from matejchain.chat.gpt import GPT
from matejchain.conv_hist import ConvTurnHist
from matejchain.msg import SystemMsg, UserMsg
from matejchain.tools import ToolBase
from matejchain.tools.tool_executor import DEFAULT_TOOL_ERROR_TEMPLATE, ToolExecutor


class Agent:
    def __init__(
        self,
        gpt: GPT,
        system_msg: SystemMsg,
        tools: list[ToolBase] | None = None,
        conv_hist_limit=5,
        tool_call_iter_limit=1,
        tool_err_template=DEFAULT_TOOL_ERROR_TEMPLATE,
    ):
        self.gpt = gpt
        self.tool_executor = ToolExecutor(tools=tools, error_template=tool_err_template)
        self.conv_hist = ConvTurnHist(system_msg=system_msg, max_turns=conv_hist_limit)
        self.tools = tools
        self.max_iter = tool_call_iter_limit

    def chat(self, user_msg: UserMsg):
        self.conv_hist.add(user_msg)
        assistant_msg = self.gpt.generate(messages=self.conv_hist.get(), tools=self.tools, n=1)
        n_iter = 0
        self.conv_hist.add(assistant_msg)
        while (assistant_msg.tool_calls is not None) and n_iter < self.max_iter:
            n_iter += 1
            assistant_msg = self.gpt.generate(messages=self.conv_hist.get(), tools=self.tools, n=1)
