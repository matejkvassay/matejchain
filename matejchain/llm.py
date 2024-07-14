from matejchain.msg import Msg, AssMsg
from matejchain.chat_hist import ChatHist
from matejchain.tool_base import ToolBase
from openai import OpenAI
from openai.types.chat import ChatCompletion
from matejchain.tool_call_req import ToolCallReq

client = OpenAI()


class LLM:
    """
    Provides functionality generate text based on input messages.
    LLM can either both generate responses or make choice of a tool call.
    """

    def __init__(self, model: str):
        """
        :param model: str name of OpenAI model, e.g. "gpt-3.5-turbo"
        """
        self.model = model
        self.client = OpenAI()

    def generate_one(
        self, msgs: list[Msg] | ChatHist, tools: list[ToolBase] | None = None, **api_kwargs
    ) -> AssMsg | list[ToolCallReq]:
        """
        Generate one text response with LLM.

        If you pass list of ToolBase instances into "tools" argument, you may expect
        response to be either AssMsg instance (if LLM decided no tool calls are necessary)
        or list of ToolCallReq (if LLM decided tool calls are needed).

        :param msgs: List of Msg instances - llm generation input.
                     All of these will be provided to LLM via completions API.
        :param tools: list of TooBase instances - tools to be passed to completions API
        :param api_kwargs: Optional keyword args to be passed as completions API args
                           (e.g. temperature=0.5, seed=420).
                           See https://platform.openai.com/docs/api-reference/chat/create
                           for more details.
        :return: Either AssMsg instance or list of ToolCallReq in case tools were provided
                 and subsequently tool call predicted by LLM.
        """
        return self.generate(msgs, 1, tools, **api_kwargs)[0]

    def generate(
        self,
        msgs: list[Msg] | ChatHist,
        choices: int = 1,
        tools: list[ToolBase] | None = None,
        **api_kwargs,
    ) -> list[AssMsg | list[ToolCallReq]]:
        """
        Generate multiple text responses with LLM for single given messages list input.

        If you pass list of ToolBase instances into "tools" argument, you may expect
        responses to be either AssMsg instance (if LLM decided no tool calls are necessary)
        or list of ToolCallReq (if LLM decided tool calls are needed).

        :param msgs: List of Msg instances - llm generation input.
                     All of these will be provided to LLM via completions API.
        :param tools: list of TooBase instances - tools to be passed to completions API
        :param choices: int number of llm generations to make, default is 1.
        :param api_kwargs: Optional keyword args to be passed as completions
                           API args (e.g. temperature=0.5, seed=420)
                           See https://platform.openai.com/docs/api-reference/chat/create
                           for more details.
        :return: List of responses. Responses are either AssMsg instance or list of ToolCallReq
                 in case tools were provided and subsequently tool call predicted by LLM.
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
        return self._parse_completions_response(completion)

    @staticmethod
    def _parse_completions_response(completion: ChatCompletion) -> list[AssMsg | ToolCallReq]:
        """
        Parses ChatCompletion choices into list of assistant responses and tools.
        :param completion: ChatCompletion OpenAI object
        :return: list of responses, response is either AssMsg or ToolCallReq
        """
        results = []
        for c in completion.choices:
            if c.message.tool_calls is None:
                results.append(AssMsg(c.message.content))
            else:
                results.append([ToolCallReq(t) for t in c.message.tool_calls])
        return results
