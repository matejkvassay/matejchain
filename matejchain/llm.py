from matejchain.message import Message, AssistantMessage
from matejchain.chat_history import ChatHistory
from matejchain.tool_base import ToolBase
from openai import OpenAI
from openai.types.chat import ChatCompletion
from matejchain.tool_call_req import ToolCallReq
from typing import Iterable

client = OpenAI()


class LLM:
    """
    Provides functionality generate text based on input messages.
    LLM can either both generate responses or make choice of a tool call.
    """

    def __init__(self, model: str):
        """
        :param model: str name of OpenAI model, e.g. "gpt-4o-mini"
        """
        self.model = model
        self.client = OpenAI()

    def generate_one(
        self,
        messages: Iterable[Message] | ChatHistory,
        tools: Iterable[ToolBase] | None = None,
        **api_kwargs,
    ) -> tuple[Message, list[ToolCallReq] | None]:
        return self.generate(messages, 1, tools, **api_kwargs)[0]

    def generate(
        self,
        messages: Iterable[Message] | ChatHistory,
        choices: int = 1,
        tools: list[ToolBase] | None = None,
        **api_kwargs,
    ):
        if isinstance(messages, ChatHistory):
            messages = messages.to_openai()
        else:
            messages = [m.to_openai() for m in messages]

        if tools is not None:
            api_kwargs = api_kwargs.copy()
            api_kwargs["tools"] = [t.to_openai() for t in tools]

        completion = self.client.chat.completions.create(
            model=self.model, messages=messages, n=choices, **api_kwargs
        )
        return self._parse_completions_response(completion)

    @staticmethod
    def _parse_completions_response(completion: ChatCompletion) -> list[AssMsg | ToolCallReq]:
        results = []
        for c in completion.choices:
            if c.message.tool_calls is None:
                results.append(AssistantMessage(c.message))
            else:
                results.append([ToolCallReq(t) for t in c.message.tool_calls])
        return results
