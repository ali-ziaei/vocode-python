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

    async def terminate_audio_service(self, conversation_id: str):
        await self.clear_cache(conversation_id)
