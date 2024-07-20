from pydantic import BaseModel
from matejchain.tool_call_request import ToolCallRequest
from matejchain.message import AssistantMessage


class LLMResponse(BaseModel):
    message: AssistantMessage
    tool_calls: list[ToolCallRequest]
