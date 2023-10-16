from __future__ import annotations

import asyncio
import datetime
import json
import logging
import queue
import random
import threading
import time
import typing
from typing import Any, Awaitable, Callable, Generic, Optional, Tuple, TypeVar, cast

from vocode.streaming.action.worker import ActionsWorker
from vocode.streaming.agent.base_agent import (
    AgentInput,
    AgentResponse,
    AgentResponseFillerAudio,
    AgentResponseMessage,
    AgentResponseStop,
    AgentResponseType,
    BaseAgent,
    TranscriptionAgentInput,
)
from vocode.streaming.agent.bot_sentiment_analyser import BotSentimentAnalyser
from vocode.streaming.agent.chat_gpt_agent import ChatGPTAgent
from vocode.streaming.audio.base_audio_service import BaseThreadAsyncAudioService
from vocode.streaming.constants import (
    ALLOWED_IDLE_TIME,
    PER_CHUNK_ALLOWANCE_SECONDS,
)
from vocode.streaming.models.actions import ActionInput
from vocode.streaming.models.agent import ChatGPTAgentConfig, FillerAudioConfig
from vocode.streaming.models.audio import AudioServiceConfig
from vocode.streaming.models.events import Sender, FillerEvent, SpokenMetaData
from vocode.streaming.models.log_message import BaseLog
from vocode.streaming.models.message import BaseMessage, SSMLMessage
from vocode.streaming.models.synthesizer import SentimentConfig
from vocode.streaming.models.transcriber import EndpointingConfig, TranscriberConfig
from vocode.streaming.models.transcript import (
    Message,
    Transcript,
    TranscriptCompleteEvent,
)
from vocode.streaming.output_device.base_output_device import BaseOutputDevice
from vocode.streaming.synthesizer.base_synthesizer import (
    BaseSynthesizer,
    FillerAudio,
    SynthesisResult,
)
from vocode.streaming.transcriber.base_transcriber import BaseTranscriber, Transcription
from vocode.streaming.utils import create_conversation_id, get_chunk_size_per_second
from vocode.streaming.utils.conversation_logger_adapter import wrap_logger
from vocode.streaming.utils.events_manager import EventsManager
from vocode.streaming.utils.goodbye_model import GoodbyeModel
from vocode.streaming.utils.state_manager import ConversationStateManager
from vocode.streaming.utils.worker import (
    AsyncQueueWorker,
    InterruptibleAgentResponseEvent,
    InterruptibleAgentResponseWorker,
    InterruptibleEvent,
    InterruptibleEventFactory,
    InterruptibleWorker,
)

OutputDeviceType = TypeVar("OutputDeviceType", bound=BaseOutputDevice)


class StreamingConversation(Generic[OutputDeviceType]):
    class QueueingInterruptibleEventFactory(InterruptibleEventFactory):
        def __init__(self, conversation: "StreamingConversation"):
            self.conversation = conversation

        def create_interruptible_event(
            self, payload: Any, is_interruptible: bool = True
        ) -> InterruptibleEvent[Any]:
            interruptible_event: InterruptibleEvent = (
                super().create_interruptible_event(payload, is_interruptible)
            )
            self.conversation.interruptible_events.put_nowait(interruptible_event)
            return interruptible_event

        def create_interruptible_agent_response_event(
            self,
            payload: Any,
            is_interruptible: bool = True,
            agent_response_tracker: Optional[asyncio.Event] = None,
        ) -> InterruptibleAgentResponseEvent:
            interruptible_event = super().create_interruptible_agent_response_event(
                payload,
                is_interruptible=is_interruptible,
                agent_response_tracker=agent_response_tracker,
            )
            self.conversation.interruptible_events.put_nowait(interruptible_event)
            return interruptible_event

    class AudioServiceWorker(AsyncQueueWorker):
        def __init__(
            self,
            input_queue: asyncio.Queue[bytes],
            output_queue: asyncio.Queue[bytes],
            conversation: "StreamingConversation",
        ):
            super().__init__(input_queue, output_queue)
            self.input_queue = input_queue
            self.output_queue = output_queue
            self.conversation = conversation

        async def publish_ask_more_time_filler(self):
            # when agent or customer are speaking, we return
            if not self.conversation.spoken_metadata.ready_to_publish_filler:
                return

            if not self.conversation.agent_filler_config.ask_more_time:
                return

            current_time = time.time()
            filler_phrase = None

            if self.conversation.spoken_metadata.customer_last_spoken_end_time is None:
                return

            if self.conversation.spoken_metadata.agent_last_spoken_end_time is None:
                return

            # case 1: agent need more time (we make sure agent is spoken last)
            if (
                self.conversation.spoken_metadata.customer_last_spoken_end_time
                > self.conversation.spoken_metadata.agent_last_spoken_end_time
                and current_time
                - self.conversation.spoken_metadata.customer_last_spoken_end_time
                > self.conversation.agent_filler_config.ask_more_time.threshold_sec
            ):
                filler_phrase = random.choice(
                    self.conversation.agent_filler_config.ask_more_time.filler_phrases
                )

                filler_ssml = (
                    filler_phrase
                    + f'<break time="{self.conversation.agent_filler_config.ask_more_time.trailing_silence}s"/>'
                )
                await self.publish_filler(
                    SSMLMessage(text=filler_phrase, ssml=filler_ssml)
                )

        async def publish_ask_speak_up_filler(self):
            if not self.conversation.spoken_metadata.ready_to_publish_filler:
                return

            if not self.conversation.agent_filler_config.speak_up:
                return

            if self.conversation.spoken_metadata.agent_last_spoken_end_time is None:
                return

            current_time = time.time()
            filler_phrase = None

            # case 2: agent ask customer to speak
            # if customer has not spoken at all (after greeting)
            if self.conversation.spoken_metadata.customer_last_spoken_end_time is None:
                if (
                    current_time
                    - self.conversation.spoken_metadata.agent_last_spoken_end_time
                    > self.conversation.agent_filler_config.speak_up.threshold_sec
                ):
                    filler_phrase = random.choice(
                        self.conversation.agent_filler_config.speak_up.filler_phrases
                    )
                    await self.publish_filler(BaseMessage(text=filler_phrase))
                    return

            if (
                current_time
                - self.conversation.spoken_metadata.agent_last_spoken_end_time
                > self.conversation.agent_filler_config.speak_up.threshold_sec
                and self.conversation.spoken_metadata.agent_last_spoken_end_time
                > self.conversation.spoken_metadata.customer_last_spoken_end_time
            ):
                filler_phrase = random.choice(
                    self.conversation.agent_filler_config.speak_up.filler_phrases
                )
                await self.publish_filler(BaseMessage(text=filler_phrase))
                return

        async def publish_filler(self, filler_message: BaseMessage):
            self.conversation.events_manager.publish_event(
                FillerEvent(
                    conversation_id=self.conversation.id,
                    filler_message=filler_message,
                )
            )
            self.conversation.agent.produce_interruptible_agent_response_event_nonblocking(
                AgentResponseMessage(message=filler_message),
                is_interruptible=self.conversation.agent.agent_config.allow_agent_to_be_cut_off,
            )
            filler_log = BaseLog(
                conversation_id=self.conversation.id,
                message="FILLER: Published filler.",
                time_stamp=datetime.datetime.utcnow(),
                text=filler_message.text,
            )
            self.conversation.logger.debug(json.dumps(filler_log.to_dict()))
            self.conversation.spoken_metadata.ready_to_publish_filler = False

        async def process(self, item: bytes):
            self.output_queue.put_nowait(item)
            await self.publish_ask_more_time_filler()
            await self.publish_ask_speak_up_filler()

    class TranscriptionsWorker(AsyncQueueWorker):
        """Processes all transcriptions: sends an interrupt if needed
        and sends final transcriptions to the output queue"""

        def __init__(
            self,
            input_queue: asyncio.Queue[Transcription],
            output_queue: asyncio.Queue[InterruptibleEvent[AgentInput]],
            conversation: "StreamingConversation",
            interruptible_event_factory: InterruptibleEventFactory,
        ):
            super().__init__(input_queue, output_queue)
            self.input_queue = input_queue
            self.output_queue = output_queue
            self.conversation = conversation
            self.interruptible_event_factory = interruptible_event_factory
            self.start_speaking = False

        async def process(self, transcription: Transcription):
            self.conversation.mark_last_action_timestamp()
            if transcription.message.strip() == "":
                self.conversation.logger.info("Ignoring empty transcription")
                return
            if transcription.is_final:
                self.start_speaking = False
                self.conversation.spoken_metadata.customer_last_spoken_end_time = (
                    time.time()
                )
                self.conversation.spoken_metadata.ready_to_publish_filler = True

                asr_log = BaseLog(
                    conversation_id=self.conversation.id,
                    message="ASR: Final transcription.",
                    time_stamp=datetime.datetime.utcnow(),
                    text=f'Transcription: "{transcription.message}", Latency: "{transcription.latency}" seconds.',
                )
                self.conversation.logger.debug(json.dumps(asr_log.to_dict()))
            else:
                if not self.start_speaking:
                    self.conversation.spoken_metadata.customer_last_spoken_start_time = (
                        time.time()
                    )
                self.start_speaking = True
                self.conversation.spoken_metadata.customer_last_spoken_end_time = None
                self.conversation.spoken_metadata.ready_to_publish_filler = False

                asr_log = BaseLog(
                    conversation_id=self.conversation.id,
                    message="ASR: Partial transcription.",
                    time_stamp=datetime.datetime.utcnow(),
                    text=f'Transcription: "{transcription.message}", Latency: "{transcription.latency}" seconds.',
                )
                self.conversation.logger.debug(json.dumps(asr_log.to_dict()))

            if (
                not self.conversation.is_human_speaking
                and self.conversation.is_interrupt(transcription)
            ):
                self.conversation.current_transcription_is_interrupt = (
                    self.conversation.broadcast_interrupt()
                )
                if self.conversation.current_transcription_is_interrupt:
                    self.conversation.logger.debug("sending interrupt")
                base_log = BaseLog(
                    conversation_id=self.conversation.id,
                    message="Human started speaking",
                    time_stamp=datetime.datetime.utcnow(),
                )
                self.conversation.logger.debug(json.dumps(base_log.to_dict()))

            transcription.is_interrupt = (
                self.conversation.current_transcription_is_interrupt
            )
            self.conversation.is_human_speaking = not transcription.is_final
            if transcription.is_final:
                # we use getattr here to avoid the dependency cycle between VonageCall and StreamingConversation
                event = self.interruptible_event_factory.create_interruptible_event(
                    TranscriptionAgentInput(
                        transcription=transcription,
                        conversation_id=self.conversation.id,
                        vonage_uuid=getattr(self.conversation, "vonage_uuid", None),
                        twilio_sid=getattr(self.conversation, "twilio_sid", None),
                    )
                )
                self.output_queue.put_nowait(event)

    class FillerAudioWorker(InterruptibleAgentResponseWorker):
        """
        - Waits for a configured number of seconds and then sends filler audio to the output
        - Exposes wait_for_filler_audio_to_finish() which the AgentResponsesWorker waits on before
          sending responses to the output queue
        """

        def __init__(
            self,
            input_queue: asyncio.Queue[InterruptibleAgentResponseEvent[FillerAudio]],
            conversation: "StreamingConversation",
        ):
            super().__init__(input_queue=input_queue)
            self.input_queue = input_queue
            self.conversation = conversation
            self.current_filler_seconds_per_chunk: Optional[int] = None
            self.filler_audio_started_event: Optional[threading.Event] = None

        async def wait_for_filler_audio_to_finish(self):
            if (
                self.filler_audio_started_event is None
                or not self.filler_audio_started_event.set()
            ):
                self.conversation.logger.debug(
                    "Not waiting for filler audio to finish since we didn't send any chunks"
                )
                return
            if self.interruptible_event and isinstance(
                self.interruptible_event, InterruptibleAgentResponseEvent
            ):
                await self.interruptible_event.agent_response_tracker.wait()

        def interrupt_current_filler_audio(self):
            return self.interruptible_event and self.interruptible_event.interrupt()

        async def process(self, item: InterruptibleAgentResponseEvent[FillerAudio]):
            try:
                filler_audio = item.payload
                assert self.conversation.filler_audio_config is not None
                filler_synthesis_result = filler_audio.create_synthesis_result()
                self.current_filler_seconds_per_chunk = filler_audio.seconds_per_chunk
                silence_threshold = (
                    self.conversation.filler_audio_config.silence_threshold_seconds
                )
                await asyncio.sleep(silence_threshold)
                self.conversation.logger.debug("Sending filler audio to output")
                self.filler_audio_started_event = threading.Event()

                self.conversation.spoken_metadata.agent_last_spoken_start_time = (
                    time.time()
                )
                self.conversation.spoken_metadata.agent_last_spoken_end_time = None
                self.conversation.spoken_metadata.ready_to_publish_filler = False
                await self.conversation.send_speech_to_output(
                    filler_audio.message.text,
                    filler_synthesis_result,
                    item.interruption_event,
                    filler_audio.seconds_per_chunk,
                    started_event=self.filler_audio_started_event,
                )
                self.conversation.spoken_metadata.agent_last_spoken_end_time = (
                    time.time()
                )
                self.conversation.spoken_metadata.ready_to_publish_filler = True
                item.agent_response_tracker.set()
            except asyncio.CancelledError:
                pass

    class AgentResponsesWorker(InterruptibleAgentResponseWorker):
        """Runs Synthesizer.create_speech and sends the SynthesisResult to the output queue"""

        def __init__(
            self,
            input_queue: asyncio.Queue[InterruptibleAgentResponseEvent[AgentResponse]],
            output_queue: asyncio.Queue[
                InterruptibleAgentResponseEvent[Tuple[BaseMessage, SynthesisResult]]
            ],
            conversation: "StreamingConversation",
            interruptible_event_factory: InterruptibleEventFactory,
        ):
            super().__init__(
                input_queue=input_queue,
                output_queue=output_queue,
            )
            self.input_queue = input_queue
            self.output_queue = output_queue
            self.conversation = conversation
            self.interruptible_event_factory = interruptible_event_factory
            self.chunk_size = int(
                get_chunk_size_per_second(
                    self.conversation.synthesizer.get_synthesizer_config().audio_encoding,
                    self.conversation.synthesizer.get_synthesizer_config().sampling_rate,
                )
                * self.conversation.synthesizer.get_synthesizer_config().text_to_speech_chunk_size_seconds,
            )

        def send_filler_audio(self, agent_response_tracker: Optional[asyncio.Event]):
            assert self.conversation.filler_audio_worker is not None
            self.conversation.logger.debug("Sending filler audio")
            if self.conversation.synthesizer.filler_audios:
                filler_audio = random.choice(
                    self.conversation.synthesizer.filler_audios
                )
                self.conversation.logger.debug(f"Chose {filler_audio.message.text}")
                event = self.interruptible_event_factory.create_interruptible_agent_response_event(
                    filler_audio,
                    is_interruptible=filler_audio.is_interruptible,
                    agent_response_tracker=agent_response_tracker,
                )
                self.conversation.filler_audio_worker.consume_nonblocking(event)
            else:
                self.conversation.logger.debug(
                    "No filler audio available for synthesizer"
                )

        async def process(self, item: InterruptibleAgentResponseEvent[AgentResponse]):
            if not self.conversation.synthesis_enabled:
                self.conversation.logger.debug(
                    "Synthesis disabled, not synthesizing speech"
                )
                return
            try:
                agent_response = item.payload
                if isinstance(agent_response, AgentResponseFillerAudio):
                    self.send_filler_audio(item.agent_response_tracker)
                    return
                if isinstance(agent_response, AgentResponseStop):
                    self.conversation.logger.debug("Agent requested to stop")
                    item.agent_response_tracker.set()
                    await self.conversation.terminate()
                    return

                agent_response_message = typing.cast(
                    AgentResponseMessage, agent_response
                )

                if self.conversation.filler_audio_worker is not None:
                    if (
                        self.conversation.filler_audio_worker.interrupt_current_filler_audio()
                    ):
                        await self.conversation.filler_audio_worker.wait_for_filler_audio_to_finish()

                tts_log = BaseLog(
                    conversation_id=self.conversation.id,
                    message="TTS: Synthesizing speech.",
                    time_stamp=datetime.datetime.utcnow(),
                    text=agent_response_message.message.text,
                )
                self.conversation.logger.debug(json.dumps(tts_log.to_dict()))

                synthesis_result = await self.conversation.synthesizer.create_speech(
                    agent_response_message.message,
                    self.chunk_size,
                    bot_sentiment=self.conversation.bot_sentiment,
                    conversation_id=self.conversation.id,
                )
                self.produce_interruptible_agent_response_event_nonblocking(
                    (agent_response_message.message, synthesis_result),
                    is_interruptible=item.is_interruptible,
                    agent_response_tracker=item.agent_response_tracker,
                )
            except asyncio.CancelledError:
                pass

    class SynthesisResultsWorker(InterruptibleAgentResponseWorker):
        """Plays SynthesisResults from the output queue on the output device"""

        def __init__(
            self,
            input_queue: asyncio.Queue[
                InterruptibleAgentResponseEvent[Tuple[BaseMessage, SynthesisResult]]
            ],
            conversation: "StreamingConversation",
        ):
            super().__init__(input_queue=input_queue)
            self.input_queue = input_queue
            self.conversation = conversation

        async def process(
            self,
            item: InterruptibleAgentResponseEvent[Tuple[BaseMessage, SynthesisResult]],
        ):
            try:
                message, synthesis_result = item.payload
                # create an empty transcript message and attach it to the transcript
                transcript_message = Message(
                    text="",
                    sender=Sender.BOT,
                )
                self.conversation.transcript.add_message(
                    message=transcript_message,
                    conversation_id=self.conversation.id,
                    publish_to_events_manager=False,
                )
                self.conversation.spoken_metadata.agent_last_spoken_start_time = (
                    time.time()
                )
                self.conversation.spoken_metadata.agent_last_spoken_end_time = None
                self.conversation.spoken_metadata.ready_to_publish_filler = False
                message_sent, cut_off = await self.conversation.send_speech_to_output(
                    message.text,
                    synthesis_result,
                    item.interruption_event,
                    self.conversation.synthesizer.get_synthesizer_config().text_to_speech_chunk_size_seconds,
                    transcript_message=transcript_message,
                )
                self.conversation.spoken_metadata.agent_last_spoken_end_time = (
                    time.time()
                )
                self.conversation.spoken_metadata.ready_to_publish_filler = True
                # publish the transcript message now that it includes what was said during send_speech_to_output
                self.conversation.transcript.maybe_publish_transcript_event_from_message(
                    message=transcript_message,
                    conversation_id=self.conversation.id,
                )
                item.agent_response_tracker.set()
                tts_log = BaseLog(
                    conversation_id=self.conversation.id,
                    message="TTS: Message played back.",
                    time_stamp=datetime.datetime.utcnow(),
                    text=message_sent,
                )
                self.conversation.logger.debug(json.dumps(tts_log.to_dict()))
                if cut_off:
                    await self.conversation.agent.update_last_bot_message_on_cut_off(
                        message_sent, conversation_id=self.conversation.id
                    )

                if self.conversation.agent.agent_config.end_conversation_on_goodbye:
                    goodbye_detected_task = (
                        self.conversation.agent.create_goodbye_detection_task(
                            message_sent
                        )
                    )
                    try:
                        if await asyncio.wait_for(goodbye_detected_task, 0.1):
                            self.conversation.logger.debug(
                                "Agent said goodbye, ending call"
                            )
                            await self.conversation.terminate()
                    except asyncio.TimeoutError:
                        pass
            except asyncio.CancelledError:
                pass

    def __init__(
        self,
        output_device: OutputDeviceType,
        audio_service: BaseThreadAsyncAudioService[AudioServiceConfig],
        transcriber: BaseTranscriber[TranscriberConfig],
        agent: BaseAgent,
        synthesizer: BaseSynthesizer,
        conversation_id: Optional[str] = None,
        per_chunk_allowance_seconds: float = PER_CHUNK_ALLOWANCE_SECONDS,
        events_manager: Optional[EventsManager] = None,
        echo_mode: Optional[bool] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.id = conversation_id or create_conversation_id()
        self.logger = wrap_logger(
            logger or logging.getLogger(__name__),
            conversation_id=self.id,
        )

        self.echo_mode = echo_mode
        self.output_device = output_device
        self.audio_service = audio_service
        self.transcriber = transcriber
        self.agent = agent

        # initiate filler pause tracking
        self.spoken_metadata = SpokenMetaData()

        self.agent_filler_config = self.agent.get_agent_config().agent_filler_config

        self.synthesizer = synthesizer
        self.synthesis_enabled = True

        self.interruptible_events: queue.Queue[InterruptibleEvent] = queue.Queue()
        self.interruptible_event_factory = self.QueueingInterruptibleEventFactory(
            conversation=self
        )
        self.agent.set_interruptible_event_factory(self.interruptible_event_factory)
        self.synthesis_results_queue: asyncio.Queue[
            InterruptibleAgentResponseEvent[Tuple[BaseMessage, SynthesisResult]]
        ] = asyncio.Queue()
        self.filler_audio_queue: asyncio.Queue[
            InterruptibleAgentResponseEvent[FillerAudio]
        ] = asyncio.Queue()
        self.state_manager = self.create_state_manager()

        self.audio_service_worker = self.AudioServiceWorker(
            input_queue=self.audio_service.output_queue,
            output_queue=self.transcriber.input_queue,
            conversation=self,
        )

        self.transcriptions_worker = self.TranscriptionsWorker(
            input_queue=self.transcriber.output_queue,
            output_queue=self.agent.get_input_queue(),
            conversation=self,
            interruptible_event_factory=self.interruptible_event_factory,
        )
        self.agent.attach_conversation_state_manager(self.state_manager)
        self.agent_responses_worker = self.AgentResponsesWorker(
            input_queue=self.agent.get_output_queue(),
            output_queue=self.synthesis_results_queue,
            conversation=self,
            interruptible_event_factory=self.interruptible_event_factory,
        )
        self.actions_worker = None
        if self.agent.get_agent_config().actions:
            self.actions_worker = ActionsWorker(
                input_queue=self.agent.actions_queue,
                output_queue=self.agent.get_input_queue(),
                interruptible_event_factory=self.interruptible_event_factory,
                action_factory=self.agent.action_factory,
            )
            self.actions_worker.attach_conversation_state_manager(self.state_manager)
        self.synthesis_results_worker = self.SynthesisResultsWorker(
            input_queue=self.synthesis_results_queue, conversation=self
        )
        self.filler_audio_worker = None
        self.filler_audio_config: Optional[FillerAudioConfig] = None
        if self.agent.get_agent_config().send_filler_audio:
            self.filler_audio_worker = self.FillerAudioWorker(
                input_queue=self.filler_audio_queue, conversation=self
            )

        self.events_manager = events_manager or EventsManager()
        self.events_task: Optional[asyncio.Task] = None
        self.per_chunk_allowance_seconds = per_chunk_allowance_seconds
        self.transcript = Transcript()
        self.transcript.attach_events_manager(self.events_manager)
        self.bot_sentiment = None
        if self.agent.get_agent_config().track_bot_sentiment:
            self.sentiment_config = (
                self.synthesizer.get_synthesizer_config().sentiment_config
            )
            if not self.sentiment_config:
                self.sentiment_config = SentimentConfig()
            self.bot_sentiment_analyser = BotSentimentAnalyser(
                emotions=self.sentiment_config.emotions
            )

        self.is_human_speaking = False
        self.active = False
        self.mark_last_action_timestamp()

        self.check_for_idle_task: Optional[asyncio.Task] = None
        self.track_bot_sentiment_task: Optional[asyncio.Task] = None

        self.current_transcription_is_interrupt: bool = False

        # tracing
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

    def create_state_manager(self) -> ConversationStateManager:
        return ConversationStateManager(conversation=self)

    async def start(self, mark_ready: Optional[Callable[[], Awaitable[None]]] = None):
        if self.echo_mode:
            self.active = True
            if len(self.events_manager.subscriptions) > 0:
                self.events_task = asyncio.create_task(self.events_manager.start())
            self.logger.debug("Running echo mode (None of services running)")
            return

        self.audio_service.start()
        self.transcriber.start()
        self.audio_service_worker.start()
        self.transcriptions_worker.start()
        self.agent_responses_worker.start()
        self.synthesis_results_worker.start()
        self.output_device.start()
        if self.filler_audio_worker is not None:
            self.filler_audio_worker.start()
        if self.actions_worker is not None:
            self.actions_worker.start()
        is_ready = await self.transcriber.ready()
        if not is_ready:
            raise Exception("Transcriber startup failed")
        if self.agent.get_agent_config().send_filler_audio:
            if not isinstance(
                self.agent.get_agent_config().send_filler_audio, FillerAudioConfig
            ):
                self.filler_audio_config = FillerAudioConfig()
            else:
                self.filler_audio_config = typing.cast(
                    FillerAudioConfig, self.agent.get_agent_config().send_filler_audio
                )
            await self.synthesizer.set_filler_audios(self.filler_audio_config)

        self.agent.start()
        initial_message = self.agent.get_agent_config().initial_message
        if initial_message:
            asyncio.create_task(self.send_initial_message(initial_message))
        self.agent.attach_transcript(self.transcript)
        if mark_ready:
            await mark_ready()
        if self.synthesizer.get_synthesizer_config().sentiment_config:
            await self.update_bot_sentiment()
        self.active = True
        if self.synthesizer.get_synthesizer_config().sentiment_config:
            self.track_bot_sentiment_task = asyncio.create_task(
                self.track_bot_sentiment()
            )
        self.check_for_idle_task = asyncio.create_task(self.check_for_idle())
        if len(self.events_manager.subscriptions) > 0:
            self.events_task = asyncio.create_task(self.events_manager.start())

    async def send_initial_message(self, initial_message: BaseMessage):
        # TODO: configure if initial message is interruptible
        self.transcriber.mute()
        initial_message_tracker = asyncio.Event()
        agent_response_event = (
            self.interruptible_event_factory.create_interruptible_agent_response_event(
                AgentResponseMessage(message=initial_message),
                is_interruptible=False,
                agent_response_tracker=initial_message_tracker,
            )
        )
        self.agent_responses_worker.consume_nonblocking(agent_response_event)
        await initial_message_tracker.wait()
        self.transcriber.unmute()

    async def check_for_idle(self):
        """Terminates the conversation after 15 seconds if no activity is detected"""
        while self.is_active():
            if time.time() - self.last_action_timestamp > (
                self.agent.get_agent_config().allowed_idle_time_seconds
                or ALLOWED_IDLE_TIME
            ):
                self.logger.debug("Conversation idle for too long, terminating")
                await self.terminate()
                return
            await asyncio.sleep(15)

    async def track_bot_sentiment(self):
        """Updates self.bot_sentiment every second based on the current transcript"""
        prev_transcript = None
        while self.is_active():
            await asyncio.sleep(1)
            if self.transcript.to_string() != prev_transcript:
                await self.update_bot_sentiment()
                prev_transcript = self.transcript.to_string()

    async def update_bot_sentiment(self):
        new_bot_sentiment = await self.bot_sentiment_analyser.analyse(
            self.transcript.to_string()
        )
        if new_bot_sentiment.emotion:
            self.logger.debug("Bot sentiment: %s", new_bot_sentiment)
            self.bot_sentiment = new_bot_sentiment

    def receive_message(self, message: str):
        transcription = Transcription(
            message=message,
            confidence=1.0,
            is_final=True,
        )
        self.transcriptions_worker.consume_nonblocking(transcription)

    def receive_audio(self, chunk: bytes):
        self.transcriber.send_audio(chunk)

    def warmup_synthesizer(self):
        self.synthesizer.ready_synthesizer()

    def mark_last_action_timestamp(self):
        self.last_action_timestamp = time.time()

    def broadcast_interrupt(self):
        """Stops all inflight events and cancels all workers that are sending output

        Returns true if any events were interrupted - which is used as a flag for the agent (is_interrupt)
        """
        num_interrupts = 0
        while True:
            try:
                interruptible_event = self.interruptible_events.get_nowait()
                if not interruptible_event.is_interrupted():
                    if interruptible_event.interrupt():
                        self.logger.debug("Interrupting event")
                        num_interrupts += 1
            except queue.Empty:
                break
        self.agent.cancel_current_task()
        self.agent_responses_worker.cancel_current_task()
        return num_interrupts > 0

    def is_interrupt(self, transcription: Transcription):
        return transcription.confidence >= (
            self.transcriber.get_transcriber_config().min_interrupt_confidence or 0
        )

    async def send_speech_to_output(
        self,
        message: str,
        synthesis_result: SynthesisResult,
        stop_event: threading.Event,
        seconds_per_chunk: float,
        transcript_message: Optional[Message] = None,
        started_event: Optional[threading.Event] = None,
    ):
        """
        - Sends the speech chunk by chunk to the output device
          - update the transcript message as chunks come in (transcript_message is always provided for non filler audio utterances)
        - If the stop_event is set, the output is stopped
        - Sets started_event when the first chunk is sent

        Importantly, we rate limit the chunks sent to the output. For interrupts to work properly,
        the next chunk of audio can only be sent after the last chunk is played, so we send
        a chunk of x seconds only after x seconds have passed since the last chunk was sent.

        Returns the message that was sent up to, and a flag if the message was cut off
        """
        if self.transcriber.get_transcriber_config().mute_during_speech:
            self.logger.debug("Muting transcriber")
            self.transcriber.mute()
        message_sent = message
        cut_off = False
        chunk_size = int(
            seconds_per_chunk
            * get_chunk_size_per_second(
                self.synthesizer.get_synthesizer_config().audio_encoding,
                self.synthesizer.get_synthesizer_config().sampling_rate,
            )
        )
        chunk_idx = 0
        seconds_spoken = 0
        async for chunk_result in synthesis_result.chunk_generator:
            start_time = time.time()
            speech_length_seconds = seconds_per_chunk * (
                len(chunk_result.chunk) / chunk_size
            )
            seconds_spoken = chunk_idx * seconds_per_chunk
            if stop_event.is_set():
                message_sent = f"{synthesis_result.get_message_up_to(seconds_spoken)}-"
                tts_log = BaseLog(
                    conversation_id=self.id,
                    message="TTS: Interrupted, stopping text to speech.",
                    time_stamp=datetime.datetime.utcnow(),
                    text=message_sent,
                )
                self.logger.debug(json.dumps(tts_log.to_dict()))

                cut_off = True
                break
            if chunk_idx == 0:
                if started_event:
                    started_event.set()
            self.output_device.consume_nonblocking(chunk_result.chunk)
            end_time = time.time()

            tts_log = BaseLog(
                conversation_id=self.id,
                message=f'TTS: Sent chunk "{chunk_idx}" with length (sec): "{speech_length_seconds}" to output device.',
                time_stamp=datetime.datetime.utcnow(),
                text=message_sent,
            )
            self.logger.debug(json.dumps(tts_log.to_dict()))

            await asyncio.sleep(
                max(
                    speech_length_seconds
                    - (end_time - start_time)
                    - self.per_chunk_allowance_seconds,
                    0,
                )
            )
            self.mark_last_action_timestamp()
            chunk_idx += 1
            seconds_spoken += seconds_per_chunk
            if transcript_message:
                transcript_message.text = synthesis_result.get_message_up_to(
                    seconds_spoken
                )
        if self.transcriber.get_transcriber_config().mute_during_speech:
            self.logger.debug("Unmuting transcriber")
            self.transcriber.unmute()
        if transcript_message:
            transcript_message.text = message_sent
        return message_sent, cut_off

    def mark_terminated(self):
        self.active = False

    async def terminate(self):
        self.mark_terminated()
        self.broadcast_interrupt()
        self.events_manager.publish_event(
            TranscriptCompleteEvent(conversation_id=self.id, transcript=self.transcript)
        )
        if self.check_for_idle_task:
            self.logger.debug("Terminating check_for_idle Task")
            self.check_for_idle_task.cancel()
        if self.track_bot_sentiment_task:
            self.logger.debug("Terminating track_bot_sentiment Task")
            self.track_bot_sentiment_task.cancel()
        if self.events_manager and self.events_task:
            self.logger.debug("Terminating events Task")
            await self.events_manager.flush()
        self.logger.debug("Tearing down synthesizer")
        await self.synthesizer.tear_down()
        self.logger.debug("Terminating agent")
        if (
            isinstance(self.agent, ChatGPTAgent)
            and self.agent.agent_config.vector_db_config
        ):
            # Shutting down the vector db should be done in the agent's terminate method,
            # but it is done here because `vector_db.tear_down()` is async and
            # `agent.terminate()` is not async.
            self.logger.debug("Terminating vector db")
            await self.agent.vector_db.tear_down()
        self.agent.terminate()
        self.logger.debug("Terminating output device")
        self.output_device.terminate()
        self.logger.debug("Terminating speech transcriber")
        self.transcriber.terminate()
        self.logger.debug("Terminating transcriptions worker")
        self.transcriptions_worker.terminate()
        self.logger.debug("Terminating final transcriptions worker")
        self.agent_responses_worker.terminate()
        self.logger.debug("Terminating synthesis results worker")
        self.synthesis_results_worker.terminate()
        if self.filler_audio_worker is not None:
            self.logger.debug("Terminating filler audio worker")
            self.filler_audio_worker.terminate()
        if self.actions_worker is not None:
            self.logger.debug("Terminating actions worker")
            self.actions_worker.terminate()
        self.logger.debug("Successfully terminated")
        self.logger.debug("Terminating audio service")
        self.audio_service.terminate()
        self.logger.debug("Terminating audio service worker")
        self.audio_service_worker.terminate()

    def is_active(self):
        return self.active
