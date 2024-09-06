from cucumbers.message import Message, AssistantMessage
from cucumbers.chat_history import ChatHistory
from cucumbers.tool_base import ToolBase
from cucumbers.tool_call_request import ToolCallRequest
from cucumbers.llm_response import LLMResponse
from openai import OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessageToolCall
from typing import Iterable
import logging
from json import loads

logger = logging.getLogger(__name__)

client = OpenAI()


class LLM:
    def __init__(self, model: str):
        self.model = model
        self.client = OpenAI()

    def generate_one(
        self,
        messages: Iterable[Message] | ChatHistory,
        tools: Iterable[ToolBase] | None = None,
        **api_kwargs,
    ) -> LLMResponse:
        return self.generate(messages, 1, tools, **api_kwargs)[0]

    def generate(
        self,
        messages: Iterable[Message] | ChatHistory,
        choices: int = 1,
        tools: list[ToolBase] | None = None,
        **api_kwargs,
    ) -> list[LLMResponse]:
        messages = [m.to_openai() for m in messages]
        if tools is not None:
            api_kwargs = api_kwargs.copy()
            api_kwargs["tools"] = [t.to_openai() for t in tools]

        completion = self.client.chat.completions.create(
            model=self.model, messages=messages, n=choices, **api_kwargs
        )
        return self._parse_completions_response(completion)

    def _parse_completions_response(self, completion: ChatCompletion):
        llm_responses = []
        for c in completion.choices:
            message = AssistantMessage(c.message)
            tool_calls = []
            if c.message.tool_calls is not None:
                tool_calls = [self._parse_tool_call(t) for t in c.message.tool_calls]
            llm_responses.append(LLMResponse(message=message, tool_calls=tool_calls))
        return llm_responses

    def _parse_tool_call(self, tool_call: ChatCompletionMessageToolCall):
        if tool_call.type != "function":
            raise NotImplementedError(
                f"Received unsupported tool call "
                f"type: {tool_call.type}, only 'function' "
                f"is supported"
            )
        logger.debug(f"Parsing JSON tool call args: {tool_call.function.arguments}")
        kwargs = loads(tool_call.function.arguments)
        return ToolCallRequest(id=tool_call.id, name=tool_call.function.name, kwargs=kwargs)
