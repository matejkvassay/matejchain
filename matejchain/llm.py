from matejchain.msg import Msg, AssMsg
from matejchain.chat_hist import ChatHist
from matejchain.tool_base import ToolBase
from openai import OpenAI
from openai.types.chat import ChatCompletion
from matejchain.tool_call_req import ToolCallReq

client = OpenAI()


class LLM:
    """
    Uses OpenAI LLM completions API client to generate text based on input messages.
    Slim wrapper around OpenAI client.
    """

    def __init__(self, model: str):
        """
        :param model: str name of OpenAI model, e.g. "gpt-3.5-turbo"
        """
        self.model = model
        self.client = OpenAI()

    def generate_one(
            self, msgs: list[Msg] | ChatHist, tools: list[ToolBase] | None = None, **api_kwargs
    ) -> tuple[AssMsg, list[ToolCallReq]]:
        """
        Generate one text response with LLM.
        :param msgs: List of Msg instances - llm generation input.
                     All of these will be provided to LLM via completions API.
        :param tools: list of TooBase instances - tools to be passed to completions API
        :param api_kwargs: Optional keyword args to be passed as completions API args
                           (e.g. temperature=0.5, seed=420).
                           See https://platform.openai.com/docs/api-reference/chat/create
                           for more details.
        :return: tuple of (assistant message, list of tool call requests)
        """
        return self.generate(msgs, 1, tools, **api_kwargs)[0]

    def generate(
            self,
            msgs: list[Msg] | ChatHist,
            choices: int = 1,
            tools: list[ToolBase] | None = None,
            **api_kwargs) -> list[tuple[AssMsg, ToolCallReq]]:
        """
        Generate multiple text responses with LLM for single given messages list input.

        :param msgs: List of Msg instances - llm generation input.
                     All of these will be provided to LLM via completions API.
        :param tools: list of TooBase instances - tools to be passed to completions API
        :param choices: int number of llm generations to make, default is 1.
        :param api_kwargs: Optional keyword args to be passed as completions
                           API args (e.g. temperature=0.5, seed=420)
                           See https://platform.openai.com/docs/api-reference/chat/create
                           for more details.
        :return: list of tuples consisting of (assistant_message, tool calls list)
        """
        if isinstance(msgs, ChatHist):
            msgs = msgs.to_openai()
        else:
            msgs = [m.to_openai() for m in msgs]
        if tools is not None:
            api_kwargs = api_kwargs.copy()
            api_kwargs["tools"] = [t.openai_fmt for t in tools]
        completion = self.client.chat.completions.create(
            model=self.model, messages=msgs, n=choices, **api_kwargs
        )
        return self._parse_completions_response(completion, n=choices)

    @staticmethod
    def _parse_completions_response(completion: ChatCompletion, n: int) -> list[tuple[AssMsg, ToolCallReq]]:
        """
        Parses ChatCompletion choices into list of assistant responses and tools.
        :param completion: completion OpenAI client object
        :param n: int take first n choices
        :return: list
        """
        results = []
        for c in completion.choices[:n]:
            msg = AssMsg(c.message.content)
            tool_calls = []
            if c.message.tool_calls is not None:
                for t in c.message.tool_calls:
                    tool_calls.append(ToolCallReq(t))
            results.append((msg, tool_calls))
        return results
