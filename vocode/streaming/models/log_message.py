import datetime
from dataclasses import dataclass
from mashumaro import DataClassDictMixin
from enum import Enum, auto
from typing import Optional, Dict


class LogType(Enum):
    BASE = "base"
    AUDIO = "audio"
    ASR = "asr"
    NLU = "nlu"
    TTS = "tts"


@dataclass
class BaseLog(DataClassDictMixin):
    conversation_id: str
    message: str
    time_stamp: datetime.datetime
    log_type: LogType


@dataclass
class AudioLog(BaseLog):
    """This log is responsible to capture audio related loggings"""


@dataclass
class ASRLog(BaseLog):
    transcript: str
    is_final: str
    start_time: datetime.datetime
    end_time: datetime.datetime


@dataclass
class NLULog(BaseLog):
    text: Optional[str] = None
    start_time: Optional[datetime.datetime] = None
    end_time: Optional[datetime.datetime] = None
    slot_dict: Optional[Dict] = None
    action_dict: Optional[Dict] = None
    current_task: Optional[str] = None
    is_final: Optional[bool] = None


@dataclass
class TTSLog(BaseLog):
    text: str
    start_time: Optional[datetime.datetime] = None
    end_time: Optional[datetime.datetime] = None
    is_cached: Optional[bool] = None
    is_final: Optional[bool] = None
