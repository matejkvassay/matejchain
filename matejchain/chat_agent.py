from matejchain.llm import LLM
from matejchain.chat_history import ChatHistory
from matejchain.message import AssistantMessage, UserMessage
from matejchain.tool_executor import ToolExecutor


class ChatAgent:
    def __init__(self, llm: LLM, chat_hist: ChatHistory, tools=None, **completion_kwargs):
        self.llm = llm
        self.hist = chat_hist
        self.tools = tools
        self.completion_kwargs = completion_kwargs
        if tools is not None:
            self.tool_executor = ToolExecutor(tools)

    def chat(self, usr_input: str | UserMessage) -> AssistantMessage:
        if isinstance(usr_input, str):
            usr_input = UserMessage(usr_input)
        self.hist.add(usr_input)

        # response = self.llm.generate_one(self.hist, tools=self.tools, **self.completion_kwargs)
        # if isinstance(response, AssistantMessage):
        #     self.hist.add(response)
        #     return response
        # else:
        #     tool_msgs = self.tool_executor.exec(response)
        #     if tool_msgs:
        #         self.hist.add_many(tool_msgs)
        # response = self.llm.generate_one(self.hist, **self.completion_kwargs)
        return response
