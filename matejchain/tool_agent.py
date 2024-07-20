from matejchain.llm import LLM
from matejchain.chat_history import ChatHistory
from matejchain.message import AssistantMessage, UserMessage, ToolMessage
from matejchain.tool_executor import ToolExecutor


class ToolAgent:
    def __init__(self, llm: LLM, chat_history: ChatHistory, tools=None, **completion_kwargs):
        self.llm = llm
        self.hist = chat_history
        self.tools = tools
        self.completion_kwargs = completion_kwargs
        if tools is not None:
            self.tool_executor = ToolExecutor(tools)

    def chat(self, user_input: UserMessage) -> list[AssistantMessage | ToolMessage]:
        self.hist.add(user_input)
        llm_messages = []
        llm_resp = self.llm.generate_one(self.hist, tools=self.tools, **self.completion_kwargs)
        self.hist.add(llm_resp.message, append_to=llm_messages)
        if llm_resp.tool_calls:
            tool_msgs = self.tool_executor.exec(llm_resp.tool_calls)
            self.hist.add_many(tool_msgs, append_to=llm_messages)
            llm_resp = self.llm.generate_one(self.hist, tools=None, **self.completion_kwargs)
            self.hist.add(llm_resp.message, append_to=llm_messages)
        return llm_messages

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"ToolAgent chat history:\n{self.hist}"
