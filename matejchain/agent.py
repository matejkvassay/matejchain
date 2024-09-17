from matejchain.gpt import GPT
from matejchain.msg import UserMsg
from matejchain.tools import ToolBase


class Agent:
    def __init__(self, gpt: GPT, tools: list[ToolBase] | None = None, tool_call_iters=1):
        self.gpt = gpt
        self.tools = tools
        self.tool_call_iters = tool_call_iters

    def chat(self, user_msg: UserMsg):
        pass
