from matejchain.messages import Msg, AssistantMsg
from matejchain.base.tool_base import ToolBase
from matejchain.tool_call_request import ToolCallRequest
from matejchain.gpt_response import GPTResponse
from openai import OpenAI
from openai.types.chat.chat_completion import Choice
from openai.types.chat import ChatCompletionMessageToolCall
import logging
from json.decoder import JSONDecodeError
from json import loads

logger = logging.getLogger(__name__)

client = OpenAI()


class LLM:
    def __init__(self, model_name: str, completion_kwargs=None):
        self.model = model_name
        self.completion_kwargs = completion_kwargs if completion_kwargs is not None else dict()
        self.client = OpenAI()

    def generate(
        self, messages: list[Msg], tools: list[ToolBase] | None = None, **completion_kwargs
    ) -> list[GPTResponse]:
        kwargs = self.completion_kwargs.copy()
        kwargs.update(completion_kwargs)
        if tools is not None:
            kwargs["tools"] = [t.dict() for t in tools]
        messages = [m.dict() for m in messages]
        completion = self.client.chat.completions.create(
            model=self.model, messages=messages, **kwargs
        )
        gpt_responses = [self._parse_choice(choice) for choice in completion.choices]
        return gpt_responses

    def _parse_choice(self, choice: Choice) -> GPTResponse:
        msg = AssistantMsg(content=choice.message.content)
        tool_calls = []
        if choice.message.tool_calls is not None:
            tool_calls = [self._parse_tool_call(t) for t in choice.message.tool_calls]
        return GPTResponse(msg=msg, tool_calls=tool_calls)

    @staticmethod
    def _parse_tool_call(tool_call: ChatCompletionMessageToolCall):
        if tool_call.type != "function":
            raise NotImplementedError(
                f"Received unsupported tool call "
                f"type: {tool_call.type}, only 'function' "
                f"is supported"
            )
        logger.debug(f"Parsing JSON tool call args: {tool_call.function.arguments}")
        try:
            tool_kwargs = loads(tool_call.function.arguments)
            error = None
        except JSONDecodeError as ex:
            error = str(ex)
            tool_kwargs = None

        return ToolCallRequest(
            id=tool_call.id, name=tool_call.function.name, kwargs=tool_kwargs, error=error
        )
