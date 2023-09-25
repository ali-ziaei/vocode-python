import logging
from typing import Optional
import audioop
import numpy as np
import soundfile
import os
from dataclasses import dataclass
from vocode.streaming.models.audio import AudioServiceConfig
from vocode.streaming.audio.audio_service import AudioService
from vocode.streaming.models.audio_encoding import AudioEncoding


@dataclass
class AudioServiceBundle:
    audio_service: AudioService
    audio_service_config: AudioServiceConfig


class AudioServiceFactory:
    ACTIVE_AUDIO_SERVICE: dict[str, AudioServiceBundle] = {}

    def create_audio_service(
        self,
        conversation_id: str,
        audio_service_config: AudioServiceConfig,
        logger: Optional[logging.Logger] = None,
    ):
        if isinstance(audio_service_config, AudioServiceConfig):
            self.ACTIVE_AUDIO_SERVICE[conversation_id] = AudioServiceBundle(
                audio_service=AudioService(audio_service_config, logger=logger),
                audio_service_config=audio_service_config,
            )
            return self.ACTIVE_AUDIO_SERVICE[conversation_id].audio_service
        raise Exception("Invalid audio service config")

    async def clear_cache(self, conversation_id: str):
        # remove the conversation data from local memory
        self.ACTIVE_AUDIO_SERVICE.pop(conversation_id, None)

    async def save_audio_data(self, conversation_id: str):
        """save chats"""
        if conversation_id not in self.ACTIVE_AUDIO_SERVICE:
            return
        audio_service = self.ACTIVE_AUDIO_SERVICE[conversation_id].audio_service
        audio_service_config = self.ACTIVE_AUDIO_SERVICE[
            conversation_id
        ].audio_service_config
        if audio_service_config.log_dir and audio_service.audio_bytes:
            if audio_service_config.audio_encoding == AudioEncoding.MULAW:
                data = audioop.ulaw2lin(audio_service.audio_bytes, 2)
            else:
                data = audio_service.audio_bytes
            audio_data = np.frombuffer(data, dtype=np.int16)
            audio_path = os.path.join(
                audio_service_config.log_dir, conversation_id + ".flac"
            )
            soundfile.write(audio_path, audio_data, audio_service_config.sampling_rate)

    async def terminate_audio_service(self, conversation_id: str):
        await self.save_audio_data(conversation_id)
        await self.clear_cache(conversation_id)
