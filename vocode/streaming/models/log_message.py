import datetime
from dataclasses import dataclass
from mashumaro import DataClassDictMixin
from enum import Enum, auto
from typing import Optional, Dict


@dataclass
class BaseLog(DataClassDictMixin):
    conversation_id: str
    message: str
    time_stamp: datetime.datetime
    text: Optional[str] = None
    start_time: Optional[datetime.datetime] = None
    end_time: Optional[datetime.datetime] = None
