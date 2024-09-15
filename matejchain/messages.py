from typing import Literal, Optional, List
from matejchain.tool_call_request import ToolCallRequest
from matejchain.base import MsgBase


class SystemMsg(MsgBase):
    content: str
    role: str = Literal["system"]


class UserMsg(MsgBase):
    content: str
    role: str = Literal["user"]


class AssistantMsg(MsgBase):
    content: Optional[str]
    role: Literal["assistant"]
    tool_calls: Optional[List[ToolCallRequest]] = None

    def api_dict(self):
        d = self.dict(include={"role", "content"})
        if self.tool_calls is not None:
            d["tool_calls"] = [t.api_dict() for t in self.tool_calls]
        return d


class ToolMsg(MsgBase):
    content: str
    tool_call_id: str
    role: str = Literal["tool"]

    def __str__(self):
        content = self.content if self.content is not None else "<no tool output>"
        return f'{self.role.upper()}:{self.tool_call_id}: "{content}"'
