import datetime
from dataclasses import dataclass
from mashumaro import DataClassDictMixin
from enum import Enum, auto
from typing import Optional


class LogType(Enum):
    BASE = "base"
    AUDIO = "audio"
    ASR = "asr"
    TTS = "tts"


@dataclass
class BaseLog(DataClassDictMixin):
    conversation_id: str
    message: str
    time_stamp: datetime.datetime
    log_type: LogType


@dataclass
class AudioLog(BaseLog):
    pass


@dataclass
class ASRLog(BaseLog):
    transcript: str
    is_final: str
    start_time: datetime.datetime
    end_time: datetime.datetime


@dataclass
class TTSLog(BaseLog):
    text: str
    start_time: datetime.datetime
    end_time: datetime.datetime
    is_cached: Optional[bool] = None
