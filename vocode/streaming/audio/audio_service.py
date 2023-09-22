import logging
from typing import Optional

from vocode.streaming.models.audio import AudioServiceConfig
from vocode.streaming.audio.base_audio_service import BaseThreadAsyncAudioService
from vocode.streaming.models.audio_encoding import AudioEncoding


class AudioService(BaseThreadAsyncAudioService[AudioServiceConfig]):
    """Audio Service"""

    def __init__(
        self,
        audio_service_config: AudioServiceConfig,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__(audio_service_config, logger)

    def process(self, chunk: bytes) -> bytes:
        """No processing"""
        self.audio_bytes += chunk
        return chunk

    def _run_loop(self):
        stream = self.generator()

        for chunk in stream:
            processed_chunk = self.process(chunk)
            self.output_janus_queue.sync_q.put_nowait(processed_chunk)

            if self._ended:
                break
