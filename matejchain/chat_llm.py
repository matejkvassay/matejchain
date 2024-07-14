from matejchain.llm import LLM
from matejchain.chat_hist import ChatHist
from matejchain.msg import UsrMsg, AssMsg
from matejchain.tool_executor import ToolExecutor


class ChatLLM:
    def __init__(self, llm: LLM, chat_hist: ChatHist, tools=None, **completion_kwargs):
        self.llm = llm
        self.hist = chat_hist
        self.tools = tools
        self.completion_kwargs = completion_kwargs
        self.tool_executor = ToolExecutor(tools)

    def chat(self, usr_input: str | UsrMsg) -> AssMsg:
        if isinstance(usr_input, str):
            usr_input = UsrMsg(usr_input)
        self.hist.add(usr_input)
        response = self.llm.generate_one(self.hist, tools=self.tools, **self.completion_kwargs)
        if isinstance(response, AssMsg):
            self.hist.add(response)
            return response
        else:
            tool_msgs = self.tool_executor.exec(response)
            if tool_msgs:
                self.hist.add_many(tool_msgs)
        response = self.llm.generate_one(usr_input, **self.completion_kwargs)
        return response
