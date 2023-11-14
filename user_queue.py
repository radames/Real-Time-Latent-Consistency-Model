from typing import Dict, Union
from uuid import UUID
from asyncio import Queue
from PIL import Image
from typing import Tuple, Union
from uuid import UUID
from asyncio import Queue
from PIL import Image

UserId = UUID

InputParams = dict

QueueContent = Dict[str, Union[Image.Image, InputParams]]

UserQueueDict = Dict[UserId, Queue[QueueContent]]

user_queue_map: UserQueueDict = {}
