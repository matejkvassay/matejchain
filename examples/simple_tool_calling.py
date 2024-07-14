from matejchain.tool_base import ToolBase
from matejchain.tool_param import ToolParam
from matejchain.llm import LLM
from matejchain.msg import SysMsg, UsrMsg, ToolMsg


class MathAddition(ToolBase):
    def __init__(self):
        params = [
            ToolParam("first_number", "first number to add", float),
            ToolParam("second_number", "second number to add", float)]
        super().__init__(
            "math_add",
            "Always use this to add 2 numbers.",
            params,
        )

    def exec(self, first_number: float, second_number: float):
        return first_number + second_number


class MathMultiplication(ToolBase):
    def __init__(self):
        params = [
            ToolParam("first_number", "first number to add", float),
            ToolParam("second_number", "second number to add", float)]
        super().__init__(
            "math_multiply",
            "Performs mathematical multiplication of 2 given numbers.",
            params,
        )

    def exec(self, first_number: float, second_number: float):
        return first_number * second_number


llm = LLM("gpt-3.5-turbo")
msgs = [
    SysMsg("You are a math genius."),
    UsrMsg("I had 1235 USD and my aunt gave me another 1234. How many dollars do I have now?"),
]
resp, tool_calls = llm.generate_one(msgs, tools=[MathAddition(), MathMultiplication()])
print(resp)
print(tool_calls)
