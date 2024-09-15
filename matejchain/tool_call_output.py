from pydantic import BaseModel
from matejchain.messages import ToolMsg
from typing import Optional
from matejchain.base import ApiCompatibleBase


class ToolCallOutput(BaseModel, ApiCompatibleBase):
    tool_message: ToolMsg
    error: Optional[str] = None

    def api_dict(self) -> dict:
        return self.tool_message.api_dict()
