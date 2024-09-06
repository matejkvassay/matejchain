from cucumbers.tool_call_request import ToolCallRequest
from cucumbers.message import AssistantMessage


class LLMResponse:
    def __init__(self, message: AssistantMessage, tool_calls: list[ToolCallRequest]):
        self.message = message
        self.tool_calls = tool_calls

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"{self.message}; tool calls: {self.tool_calls}"
