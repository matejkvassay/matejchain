from matejchain.tool_call_request import ToolCallRequest
from matejchain.messages import AssistantMsg
from pydantic import BaseModel
from typing import List, Optional


class GPTResponse(BaseModel):
    # Assistant message, can contain empty content if tool_calls
    # are specified by LLM, but not always.
    msg: AssistantMsg

    # List of tool call requests to be executed. Can be None if no tool
    # calls are specified by LLM.
    tool_calls: Optional[List[ToolCallRequest]] = None
