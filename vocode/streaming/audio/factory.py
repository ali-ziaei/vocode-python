import logging
from typing import Optional
from vocode.streaming.models.audio import AudioServiceConfig
from vocode.streaming.audio.audio_service import AudioService


class AudioServiceFactory:
    def create_audio_service(
        self,
        conversation_id: str,
        audio_service_config: AudioServiceConfig,
        logger: Optional[logging.Logger] = None,
        log_dir: Optional[str] = None,
    ):
        if isinstance(audio_service_config, AudioServiceConfig):
            return AudioService(
                conversation_id, audio_service_config, logger=logger, log_dir=log_dir
            )
        raise Exception("Invalid audio service config")
