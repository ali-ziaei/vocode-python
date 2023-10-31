from dataclasses import dataclass, asdict
from typing import Optional
import json


@dataclass
class VocodeBaseLogMessage:
    message: str
    text: Optional[str] = None

    def __str__(self):
        return json.dumps(asdict(self))


@dataclass
class VocodeLogContext:
    conversation_id: str

    def __str__(self):
        return json.dumps(asdict(self))
