from openai.types.chat import ChatCompletionMessageToolCall
from json import loads


class ToolCallReq:
    def __init__(self, tool_call: ChatCompletionMessageToolCall):
        if tool_call.type != 'function':
            raise NotImplementedError(f'Recieved unsupported tool call type: {tool_call.type}')
        self.tool_name = tool_call.function.name
        self.kwargs = loads(tool_call.function.arguments)
