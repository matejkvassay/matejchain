from cucumbers.tool_base import ToolBase
from cucumbers.tool_param import ToolParam
from cucumbers.llm import LLM
from cucumbers.message import UserMessage
from cucumbers.chat_history import ChatHistory
from cucumbers.tool_agent import ToolAgent


class MathAddition(ToolBase):
    def __init__(self):
        params = (
            ToolParam("first_number", "first number to add", float),
            ToolParam("second_number", "second number to add", float),
        )
        super().__init__("math_add", "Always use this to add 2 numbers.", params)

    def _exec(self, first_number: float, second_number: float):
        return first_number + second_number


class MathMultiplication(ToolBase):
    def __init__(self):
        params = (
            ToolParam("first_number", "first number to add", float),
            ToolParam("second_number", "second number to add", float),
        )
        super().__init__(
            "math_multiply",
            "Performs mathematical multiplication of 2 given numbers.",
            params,
        )

    def _exec(self, first_number: float, second_number: float):
        return first_number * second_number


llm = LLM("gpt-4o-mini")

hist = ChatHistory(limit=10, sys_msg="You are a math genius.")
user_input = UserMessage(
    "I had 10 USD and my aunt gave me another 5. How many dollars do I have now? "
    "Also please multiply 4x5"
)
tool_agent = ToolAgent(llm, hist, tools=[MathAddition(), MathMultiplication()])
tool_agent.chat(user_input)
print(tool_agent)
