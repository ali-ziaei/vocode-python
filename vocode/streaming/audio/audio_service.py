import logging
from typing import Optional

from vocode.streaming.models.audio import AudioServiceConfig
from vocode.streaming.audio.base_audio_service import BaseThreadAsyncAudioService


class AudioService(BaseThreadAsyncAudioService[AudioServiceConfig]):
    """Audio Service"""

    def __init__(
        self,
        conversation_id: str,
        audio_service_config: AudioServiceConfig,
        logger: Optional[logging.Logger] = None,
        log_dir: Optional[str] = None,
    ):
        super().__init__(conversation_id, audio_service_config, logger, log_dir)

    def process(self, chunk: bytes) -> bytes:
        """No processing"""
        self.audio += chunk
        return chunk

    def _run_loop(self):
        stream = self.generator()

        for chunk in stream:
            processed_chunk = self.process(chunk)
            self.output_janus_queue.sync_q.put_nowait(processed_chunk)

            if self._ended:
                break
