import logging
from typing import Optional
import queue

from vocode.streaming.models.audio import AudioServiceConfig
from vocode.streaming.audio.base_audio_service import BaseThreadAsyncAudioService


class AudioService(BaseThreadAsyncAudioService[AudioServiceConfig]):
    """Audio Service"""

    def __init__(
        self,
        audio_service_config: AudioServiceConfig,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__(audio_service_config)
        self.logger = logger

        self._ended = False
        self.is_ready = False

    def process(self, chunk):
        return chunk

    def _run_loop(self):
        stream = self.generator()

        for chunk in stream:
            processed_chunk = self.process(chunk)
            self.output_janus_queue.sync_q.put_nowait(processed_chunk)

            if self._ended:
                break

    def generator(self):
        """audio frame generator"""
        while not self._ended:
            try:
                chunk = self.input_janus_queue.sync_q.get(timeout=5)
            except queue.Empty:
                return

            if chunk is None:
                return

            data = [chunk]
            # Now consume whatever other data's still buffered.
            while True:
                try:
                    chunk = self.input_janus_queue.sync_q.get_nowait()
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break

            yield b"".join(data)

    def terminate(self):
        self._ended = True
        super().terminate()
