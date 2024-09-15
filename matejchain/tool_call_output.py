from typing import Optional

from pydantic import BaseModel

from matejchain.base import ApiCompatibleBase
from matejchain.msg.messages import ToolMsg


class ToolCallOutput(BaseModel, ApiCompatibleBase):
    tool_message: ToolMsg
    error: Optional[str] = None

    def api_dict(self) -> dict:
        return self.tool_message.api_dict()
