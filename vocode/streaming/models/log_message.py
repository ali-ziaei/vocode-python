import datetime
from dataclasses import dataclass
from mashumaro import DataClassDictMixin
from typing import Optional


@dataclass
class BaseLog(DataClassDictMixin):
    conversation_id: str
    message: str
    time_stamp: datetime.datetime
    text: Optional[str] = None
