import logging
from typing import Optional
from vocode.streaming.models.audio import AudioServiceConfig
from vocode.streaming.audio.audio_service import AudioService


class AudioServiceFactory:
    def create_audio_service(
        self,
        audio_service_config: AudioServiceConfig,
        logger: Optional[logging.Logger] = None,
    ):
        if isinstance(audio_service_config, AudioServiceConfig):
            return AudioService(audio_service_config, logger=logger)
        raise Exception("Invalid audio service config")
