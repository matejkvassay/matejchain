from matejchain.tool_call_request import ToolCallReq
from matejchain.message import ToolMsg
import logging

logger = logging.getLogger(__name__)


class ToolExecutor:
    def __init__(self, tools):
        self.tool_idx = {t.name: t for t in tools}

    def exec(self, tool_call_reqs: list[ToolCallReq]):
        tool_responses = []
        for req in tool_call_reqs:
            tool = self.tool_idx.get(req.name, None)
            if tool is not None:
                try:
                    if req.kwargs is not None:
                        resp = tool(**req.kwargs)
                    else:
                        resp = tool()
                except Exception as ex:
                    logger.exception(f"Error executing tool call request: {req}, err: {ex}")
                if resp is not None:
                    tool_responses.append(ToolMsg(resp))
            else:
                logger.exception(f"Cound not identify tool for exec req: {req}")
