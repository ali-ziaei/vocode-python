from .audio_encoding import AudioEncoding
from .model import TypedModel
from enum import Enum


class AudioServiceType(str, Enum):
    """audio service types"""

    BASE = "audio_service_base"


class AudioServiceConfig(TypedModel, type=AudioServiceType.BASE.value):
    """Audio service config"""

    sampling_rate: int
    audio_encoding: AudioEncoding
    log_dir: str
