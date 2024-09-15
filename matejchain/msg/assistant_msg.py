from typing import List, Literal, Optional

from matejchain.base import MsgBase
from matejchain.data_models.tool_call_request import ToolCallRequest


class AssistantMsg(MsgBase):
    content: Optional[str]
    role: Literal["assistant"]
    tool_calls: Optional[List[ToolCallRequest]] = None

    def api_dict(self):
        d = self.dict(include={"role", "content"})
        if self.tool_calls is not None:
            d["tool_calls"] = [t.api_dict() for t in self.tool_calls]
        return d

    def __str__(self):
        if self.tool_calls is None:
            return super().__str__()
        tool_names = ", ".join([t.name for t in self.tool_calls])
        content = self.content if self.content is not None else "<no content>"
        return f'{self.role.upper()}: "{content}" : tool_calls=[{tool_names}]'
