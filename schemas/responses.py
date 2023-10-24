from typing import Any, Dict, List, Union
from pydantic import BaseModel

class FaceResponse(BaseModel):
	response: Dict[str, Union[str, List[Dict[str, Any]]]]