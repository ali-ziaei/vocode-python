import logging
import os
from enum import Enum
from typing import Optional, TypeVar, Union
import time
import json
import datetime
from vocode.streaming.models.log_message import BaseLog
from fastapi import WebSocket
from pythonjsonlogger import jsonlogger  # type: ignore
from vocode.streaming.agent.factory import AgentFactory
from vocode.streaming.audio.factory import AudioServiceFactory
from vocode.streaming.models.agent import AgentConfig
from vocode.streaming.models.audio import AudioServiceConfig
from vocode.streaming.models.events import PhoneCallEndedEvent
from vocode.streaming.models.synthesizer import SynthesizerConfig
from vocode.streaming.models.transcriber import TranscriberConfig
from vocode.streaming.output_device.twilio_output_device import TwilioOutputDevice
from vocode.streaming.output_device.vonage_output_device import VonageOutputDevice
from vocode.streaming.streaming_conversation import StreamingConversation
from vocode.streaming.synthesizer.factory import SynthesizerFactory
from vocode.streaming.telephony.config_manager.base_config_manager import (
    BaseConfigManager,
)
from vocode.streaming.telephony.constants import DEFAULT_SAMPLING_RATE
from vocode.streaming.transcriber.factory import TranscriberFactory
from vocode.streaming.utils import create_conversation_id
from vocode.streaming.utils.conversation_logger_adapter import wrap_logger
from vocode.streaming.utils.events_manager import EventsManager

TelephonyOutputDeviceType = TypeVar(
    "TelephonyOutputDeviceType", bound=Union[TwilioOutputDevice, VonageOutputDevice]
)


class Call(StreamingConversation[TelephonyOutputDeviceType]):
    def __init__(
        self,
        from_phone: str,
        to_phone: str,
        base_url: str,
        config_manager: BaseConfigManager,
        output_device: TelephonyOutputDeviceType,
        agent_config: AgentConfig,
        audio_service_config: AudioServiceConfig,
        transcriber_config: TranscriberConfig,
        synthesizer_config: SynthesizerConfig,
        conversation_id: Optional[str] = None,
        audio_service_factory: AudioServiceFactory = AudioServiceFactory(),
        transcriber_factory: TranscriberFactory = TranscriberFactory(),
        agent_factory: AgentFactory = AgentFactory(),
        synthesizer_factory: SynthesizerFactory = SynthesizerFactory(),
        events_manager: Optional[EventsManager] = None,
        echo_mode: Optional[bool] = None,
        logger: Optional[logging.Logger] = None,
    ):
        conversation_id = conversation_id or create_conversation_id()

        if logger and events_manager and events_manager.log_dir:
            os.makedirs(events_manager.log_dir, exist_ok=True)
            format_str: str = "%(asctime)s.%03d [%(filename)s:%(lineno)s ] [%(levelname)s] ['%(message)s']"
            log_file = os.path.join(
                events_manager.log_dir, conversation_id + ".log.jsonl"
            )
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            formatter = jsonlogger.JsonFormatter(format_str)
            formatter.converter = time.gmtime
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        self.logger = wrap_logger(
            logger or logging.getLogger(__name__),
            conversation_id=conversation_id,
        )

        self.from_phone = from_phone
        self.to_phone = to_phone
        self.base_url = base_url
        self.config_manager = config_manager
        self.echo_mode = echo_mode
        super().__init__(
            output_device,
            audio_service_factory.create_audio_service(
                conversation_id, audio_service_config, logger=self.logger
            ),
            transcriber_factory.create_transcriber(
                transcriber_config, logger=self.logger
            ),
            agent_factory.create_agent(agent_config, logger=self.logger),
            synthesizer_factory.create_synthesizer(
                synthesizer_config, logger=self.logger
            ),
            conversation_id=conversation_id,
            per_chunk_allowance_seconds=0.01,
            events_manager=events_manager,
            echo_mode=echo_mode,
            logger=self.logger,
        )

    def attach_ws(self, ws: WebSocket):
        self.logger.debug("Trying to attach WS to outbound call")
        self.output_device.ws = ws
        self.logger.debug("Attached WS to outbound call")
        log = BaseLog(
            conversation_id=self.id,
            message="Base: Attached WS to outbound call.",
            time_stamp=datetime.datetime.utcnow(),
        )
        self.logger.debug(json.dumps(log.to_dict()))

    async def attach_ws_and_start(self, ws: WebSocket):
        raise NotImplementedError

    async def tear_down(self):
        self.events_manager.publish_event(PhoneCallEndedEvent(conversation_id=self.id))
        await self.terminate()
