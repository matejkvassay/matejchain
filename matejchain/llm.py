from matejchain.msg import Msg, AssMsg
from matejchain.chat_hist import ChatHist
from openai import OpenAI

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

    def generate_one(self, msgs: list[Msg] | ChatHist, **api_kwargs) -> AssMsg:
        """
        Generate one text response with LLM.
        :param msgs: List of Msg instances - llm generation input.
                     All of these will be provided to LLM via completions API.
        :param api_kwargs: Optional keyword args to be passed as completions API args
                           (e.g. temperature=0.5, seed=420).
                           See https://platform.openai.com/docs/api-reference/chat/create
                           for more details.
        :return: single assistant message instance
        """
        return self.generate(msgs, 1, **api_kwargs)[0]

    def generate(self, msgs: list[Msg] | ChatHist, choices: int = 1, **api_kwargs) -> list[AssMsg]:
        """

        :param msgs: List of Msg instances - llm generation input.
                     All of these will be provided to LLM via completions API.
        :param choices: int number of llm generations to make, by default just 1.
        :param api_kwargs: Optional keyword args to be passed as completions
                           API args (e.g. temperature=0.5, seed=420)
                           See https://platform.openai.com/docs/api-reference/chat/create
                           for more details.
        :return: list of assistant message instances
        """
        if isinstance(msgs, ChatHist):
            msgs = msgs.to_openai()
        else:
            msgs = [m.to_openai() for m in msgs]
        completion = self.client.chat.completions.create(
            model=self.model, messages=msgs, n=choices, **api_kwargs
        )
        return [AssMsg(c.message.content) for c in completion.choices]
