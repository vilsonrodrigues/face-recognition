from typing import Any, Dict, List, Union
from pydantic import BaseModel

class Payload(BaseModel):
	# adapt your payload
	...

class FaceResponse(BaseModel):
	payloads: List[List[Payload]]
	boxes: List[List[int]]