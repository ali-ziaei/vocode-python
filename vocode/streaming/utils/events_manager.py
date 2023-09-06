from __future__ import annotations

import asyncio
import os
from typing import List, Optional

from vocode.streaming.models.events import Event, EventType


class EventsManager:
    def __init__(
        self, subscriptions: List[EventType] = [], log_dir: Optional[str] = None
    ):
        self.queue: asyncio.Queue[Event] = asyncio.Queue()
        self.subscriptions = set(subscriptions)
        self.active = False
        self.log_dir = log_dir
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

    def publish_event(self, event: Event):
        if event.type in self.subscriptions:
            self.queue.put_nowait(event)

    async def start(self):
        self.active = True
        while self.active:
            try:
                event = await self.queue.get()
            except asyncio.QueueEmpty:
                await asyncio.sleep(1)
            await self.handle_event(event)

    async def handle_event(self, event: Event):
        pass

    async def flush(self):
        self.active = False
        while True:
            try:
                event = self.queue.get_nowait()
                await self.handle_event(event)
            except asyncio.QueueEmpty:
                break
