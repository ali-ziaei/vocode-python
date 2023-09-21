import logging
from typing import Optional
import asyncio
import queue
import audioop
import numpy as np
import os
import soundfile
from typing import TypeVar, Generic
from vocode.streaming.utils.worker import ThreadAsyncWorker
from vocode.streaming.models.audio_encoding import AudioEncoding
from vocode.streaming.models.audio import AudioServiceConfig

AudioServiceConfigType = TypeVar("AudioServiceConfigType", bound=AudioServiceConfig)


class AbstractAudioService(Generic[AudioServiceConfigType]):
    """Audio service abstract"""

    def __init__(self, audio_service_config: AudioServiceConfigType):
        self.audio_service_config = audio_service_config
        self.is_muted = False
        self.audio_bytes = b""

    def mute(self):
        """Mute"""
        self.is_muted = True

    def unmute(self):
        """Unmute"""
        self.is_muted = False

    def get_audio_service_config(self) -> AudioServiceConfigType:
        """get audio service config"""
        return self.audio_service_config

    async def ready(self):
        """If audio service is ready"""
        return True

    def create_silent_chunk(self, chunk_size, sample_width=2):
        """creating silence"""
        linear_audio = b"\0" * chunk_size
        if self.get_audio_service_config().audio_encoding == AudioEncoding.LINEAR16:
            return linear_audio

        if self.get_audio_service_config().audio_encoding == AudioEncoding.MULAW:
            return audioop.lin2ulaw(linear_audio, sample_width)

        raise ValueError(
            f'"{self.get_audio_service_config().audio_encoding}" is not supported!'
        )


class BaseThreadAsyncAudioService(
    AbstractAudioService[AudioServiceConfigType], ThreadAsyncWorker
):
    """Audio service"""

    def __init__(
        self,
        conversation_id: str,
        audio_service_config: AudioServiceConfigType,
        logger: Optional[logging.Logger] = None,
    ):
        self.conversation_id = conversation_id
        self.is_muted = False
        self._ended = False
        self.input_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self.output_queue: asyncio.Queue[bytes] = asyncio.Queue()
        ThreadAsyncWorker.__init__(self, self.input_queue, self.output_queue)
        AbstractAudioService.__init__(self, audio_service_config)
        self.logger = logger

    def process(self, chunk: bytes) -> bytes:
        raise NotImplementedError

    def _run_loop(self):
        raise NotImplementedError

    def send_audio(self, chunk):
        """sending audio"""
        if not self.is_muted:
            self.consume_nonblocking(chunk)
        else:
            self.consume_nonblocking(self.create_silent_chunk(len(chunk)))

    def generator(self):
        """audio frame generator"""
        while not self._ended:
            try:
                chunk = self.input_janus_queue.sync_q.get()
            except queue.Empty:
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
        if self.audio_service_config.log_dir and self.audio_bytes:
            if self.audio_service_config.audio_encoding == AudioEncoding.MULAW:
                data = audioop.ulaw2lin(self.audio_bytes, 2)
            else:
                data = self.audio_bytes
            audio_data = np.frombuffer(data, dtype=np.int16)
            audio_path = os.path.join(
                self.audio_service_config.log_dir, self.conversation_id + ".flac"
            )
            soundfile.write(
                audio_path, audio_data, self.audio_service_config.sampling_rate
            )
        ThreadAsyncWorker.terminate(self)
