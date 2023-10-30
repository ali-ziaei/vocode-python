from dataclasses import dataclass, asdict
from typing import Optional
import json


@dataclass
class BaseLogMessage:
    message: str
    text: Optional[str] = None

    def __str__(self):
        return json.dumps(asdict(self))
