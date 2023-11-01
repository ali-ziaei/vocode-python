import asyncio
import logging
import time
from typing import Any, AsyncGenerator, Optional, Tuple, Union
import wave
import aiohttp
import datetime
import json

from vocode import getenv
from vocode.streaming.synthesizer.base_synthesizer import (
    BaseSynthesizer,
    SynthesisResult,
    tracer,
)
from vocode.streaming.models.synthesizer import (
    ElevenLabsSynthesizerConfig,
    SynthesizerType,
)
from vocode.streaming.agent.bot_sentiment_analyser import BotSentiment
from vocode.streaming.models.message import BaseMessage, SSMLMessage
from vocode.streaming.utils.mp3_helper import decode_mp3
from vocode.streaming.synthesizer.miniaudio_worker import MiniaudioWorker
from vocode.streaming.models.logging import VocodeBaseLogMessage, VocodeLogContext

ADAM_VOICE_ID = "pNInz6obpgDQGcFmaJgB"
ELEVEN_LABS_BASE_URL = "https://api.elevenlabs.io/v1/"


class ElevenLabsSynthesizer(BaseSynthesizer[ElevenLabsSynthesizerConfig]):
    def __init__(
        self,
        synthesizer_config: ElevenLabsSynthesizerConfig,
        logger: Optional[logging.Logger] = None,
        aiohttp_session: Optional[aiohttp.ClientSession] = None,
    ):
        super().__init__(synthesizer_config, aiohttp_session)

        import elevenlabs

        self.elevenlabs = elevenlabs

        self.api_key = synthesizer_config.api_key or getenv("ELEVEN_LABS_API_KEY")
        self.voice_id = synthesizer_config.voice_id or ADAM_VOICE_ID
        self.stability = synthesizer_config.stability
        self.similarity_boost = synthesizer_config.similarity_boost
        self.model_id = synthesizer_config.model_id
        self.optimize_streaming_latency = synthesizer_config.optimize_streaming_latency
        self.words_per_minute = 150
        self.experimental_streaming = synthesizer_config.experimental_streaming
        self.logger = logger or logging.getLogger(__name__)

    async def cached_chunk_generator(
        self,
        cached_audio: bytes,
        chunk_size: int,
    ) -> AsyncGenerator[SynthesisResult.ChunkResult, None]:
        for i in range(0, len(cached_audio), chunk_size):
            if i + chunk_size > len(cached_audio):
                yield SynthesisResult.ChunkResult(cached_audio[i:], True)
            else:
                yield SynthesisResult.ChunkResult(
                    cached_audio[i : i + chunk_size], False
                )

    async def create_speech(
        self,
        message: BaseMessage,
        chunk_size: int,
        bot_sentiment: Optional[BotSentiment] = None,
        conversation_id: Optional[str] = None,
    ) -> SynthesisResult:
        voice = self.elevenlabs.Voice(voice_id=self.voice_id)
        if self.stability is not None and self.similarity_boost is not None:
            voice.settings = self.elevenlabs.VoiceSettings(
                stability=self.stability, similarity_boost=self.similarity_boost
            )

        text = message.ssml if isinstance(message, SSMLMessage) else message.text

        cache_key = self.get_cache_key(text)
        audio_data = self.cache.get(cache_key)

        if audio_data is not None:
            context = VocodeLogContext(conversation_id if conversation_id else "")
            log_message = VocodeBaseLogMessage(
                message="TTS: Synthesizing speech -> found in Redis.",
                text=text,
            )
            self.logger.debug(log_message, context=context)
        else:
            context = VocodeLogContext(conversation_id if conversation_id else "")
            log_message = VocodeBaseLogMessage(
                message="TTS: Synthesizing speech -> calling API.",
                text=text,
            )
            self.logger.debug(log_message, context=context)

            url = ELEVEN_LABS_BASE_URL + f"text-to-speech/{self.voice_id}"
            if self.experimental_streaming:
                url += "/stream"

            if self.optimize_streaming_latency:
                url += f"?optimize_streaming_latency={self.optimize_streaming_latency}"
            headers = {"xi-api-key": self.api_key}
            body = {
                "text": text,
                "voice_settings": voice.settings.dict() if voice.settings else None,
            }
            if self.model_id:
                body["model_id"] = self.model_id

            create_speech_span = tracer.start_span(
                f"synthesizer.{SynthesizerType.ELEVEN_LABS.value.split('_', 1)[-1]}.create_total",
            )

            session = self.aiohttp_session

            response = await session.request(
                "POST",
                url,
                json=body,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=15),
            )
            if not response.ok:
                raise Exception(
                    f"ElevenLabs API returned {response.status} status code"
                )

            context = VocodeLogContext(conversation_id if conversation_id else "")
            log_message = VocodeBaseLogMessage(
                message="TTS: Synthesizing speech, called API -> Got api response.",
                text=text,
            )
            self.logger.debug(log_message, context=context)

            if self.experimental_streaming:
                synthesis_result = SynthesisResult(
                    self.experimental_mp3_streaming_output_generator(
                        response, chunk_size, create_speech_span, message
                    ),  # should be wav
                    lambda seconds: self.get_message_cutoff_from_voice_speed(
                        message, seconds, self.words_per_minute
                    ),
                )

                context = VocodeLogContext(conversation_id if conversation_id else "")
                log_message = VocodeBaseLogMessage(
                    message="TTS: Synthesizing speech, called API, got api response -> Did audio conversion.",
                    text=text,
                )
                self.logger.debug(log_message, context=context)
                return synthesis_result
            else:
                create_speech_span.end()
                audio_data = await response.read()
                self.cache.set(cache_key, audio_data)

        # Each of the branches below use the cached audio to generate a response
        if self.experimental_streaming:
            synthesis_result = SynthesisResult(
                self.cached_chunk_generator(audio_data, chunk_size=chunk_size),
                lambda seconds: self.get_message_cutoff_from_voice_speed(
                    message, seconds, self.words_per_minute
                ),
            )

            context = VocodeLogContext(conversation_id if conversation_id else "")
            log_message = VocodeBaseLogMessage(
                message="TTS: Synthesizing speech, got from Redis -> Did audio conversion.",
                text=text,
            )
            self.logger.debug(log_message, context=context)

            return synthesis_result
        else:
            convert_span = tracer.start_span(
                f"synthesizer.{SynthesizerType.ELEVEN_LABS.value.split('_', 1)[-1]}.convert",
            )
            output_bytes_io = decode_mp3(audio_data)

            result = self.create_synthesis_result_from_wav(
                synthesizer_config=self.synthesizer_config,
                file=output_bytes_io,
                message=message,
                chunk_size=chunk_size,
            )
            convert_span.end()

            context = VocodeLogContext(conversation_id if conversation_id else "")
            log_message = VocodeBaseLogMessage(
                message="TTS: Synthesizing speech, got from Redis -> Did audio conversion.",
                text=text,
            )
            self.logger.debug(log_message, context=context)
            return result
