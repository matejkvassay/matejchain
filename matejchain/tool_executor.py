from matejchain.tool_call_request import ToolCallRequest
from matejchain.message import ToolMessage
from matejchain.tool_base import ToolBase
import logging

logger = logging.getLogger(__name__)


class ToolExecutor:
    def __init__(self, tools: list[ToolBase], failed_msg="Failed to call tool"):
        self.tool_idx = {t.name: t for t in tools}
        self.failed_msg = failed_msg

    def exec(self, tool_call_reqs: list[ToolCallRequest]):
        tool_responses = []
        for req in tool_call_reqs:
            try:
                tool = self.tool_idx.get(req.name, None)
                if req.kwargs:
                    tool_output = tool(**req.kwargs)
                else:
                    tool_output = tool()
                tool_msg = ToolMessage(
                    name=req.name, kwargs=req.kwargs, call_id=req.id, result=tool_output
                )
            except Exception:
                tool_msg = ToolMessage(
                    name=req.name, kwargs=req.kwargs, call_id=req.id, result=self.failed_msg
                )
            tool_responses.append(tool_msg)
        return tool_responses
