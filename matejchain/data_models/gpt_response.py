from typing import List, Optional

from pydantic import BaseModel

from matejchain.data_models.tool_call_request import ToolCallRequest
from matejchain.msg.messages import AssistantMsg


class GPTResponse(BaseModel):
    # Assistant message, can contain empty content if tool_calls
    # are specified by LLM, but not always.
    msg: AssistantMsg

    # List of tool call requests to be executed. Can be None if no tool
    # calls are specified by LLM.
    tool_calls: Optional[List[ToolCallRequest]] = None
