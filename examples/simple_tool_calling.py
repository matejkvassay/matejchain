from matejchain.tool_base import ToolBase
from matejchain.tool_param import ToolParam
from matejchain.llm import LLM
from matejchain.msg import SysMsg, UsrMsg, AssMsg


class MathAddition(ToolBase):
    def __init__(self):
        params = [
            ToolParam("first_number", "first number to add", float),
            ToolParam("second_number", "second number to add", float),
        ]
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
            ToolParam("second_number", "second number to add", float),
        ]
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
    UsrMsg(
        "I had 10 USD and my aunt gave me another 5. How many dollars do I have now? "
        "Also please multiply 4x5"
    ),
]
resp = llm.generate_one(msgs, tools=[MathAddition(), MathMultiplication()])
print(f"Reponse after tool prompt: {resp}")
msgs.append(AssMsg("You have $15. 4x5 is equal to 20."))
msgs.append(UsrMsg("Wow, you are a math genius!"))
resp = llm.generate_one(msgs, tools=[MathAddition(), MathMultiplication()])
print(f"response after non tool prompt: {resp}")
