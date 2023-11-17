from typing import Dict, Union
from uuid import UUID
import asyncio
from PIL import Image
from typing import Dict, Union
from PIL import Image

InputParams = dict
UserId = UUID
EventDataContent = Dict[str, InputParams]


class UserDataEvent:
    def __init__(self):
        self.data_event = asyncio.Event()
        self.data_content: EventDataContent = {}

    def update_data(self, new_data: EventDataContent):
        self.data_content = new_data
        self.data_event.set()

    async def wait_for_data(self) -> EventDataContent:
        await self.data_event.wait()
        self.data_event.clear()
        return self.data_content


UserDataEventMap = Dict[UserId, UserDataEvent]
user_data_events: UserDataEventMap = {}
