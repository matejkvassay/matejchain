from matejchain.llm import LLM
from matejchain.chat_history import ChatHistory
from matejchain.message import AssistantMessage, UserMessage
from matejchain.tool_executor import ToolExecutor


class ChatAgent:
    def __init__(self, llm: LLM, chat_history: ChatHistory, tools=None, **completion_kwargs):
        self.llm = llm
        self.hist = chat_history
        self.tools = tools
        self.completion_kwargs = completion_kwargs
        if tools is not None:
            self.tool_executor = ToolExecutor(tools)

    def chat(self, user_input: str | UserMessage) -> AssistantMessage:
        if isinstance(user_input, str):
            user_input = UserMessage(user_input)
        self.hist.add(user_input)
        llm_resp = self.llm.generate_one(self.hist, tools=self.tools,
                                         **self.completion_kwargs)
        self.hist.add(llm_resp.message)
        if llm_resp.tool_calls:
            tool_msgs = self.tool_executor.exec(llm_resp.tool_calls)
            self.hist.add_many(tool_msgs)
            llm_resp = self.llm.generate_one(self.hist, tools=None, **self.completion_kwargs)
            self.hist.add(llm_resp.message)
        return llm_resp.message
