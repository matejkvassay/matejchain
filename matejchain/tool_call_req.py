from openai.types.chat import ChatCompletionMessageToolCall
from json import loads
import logging

logger = logging.getLogger(__name__)


class ToolCallReq:
    def __init__(self, tool_call: ChatCompletionMessageToolCall):
        if tool_call.type != 'function':
            raise NotImplementedError(f'Recieved unsupported tool call type: {tool_call.type}')
        self.tool_name = tool_call.function.name
        logger.debug(f'Parsing JSON tool call args: {tool_call.function.arguments}')
        self.kwargs = loads(tool_call.function.arguments)

    def __str__(self):
        kwargs = str(self.kwargs).replace(':', '=')
        return f'ToolCallReq: {self.tool_name}:{self.kwargs}'

    def __repr__(self):
        return self.__str__()
