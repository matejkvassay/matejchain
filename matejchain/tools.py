from matejchain.tool_base import ToolBase
from matejchain.tool_param import ToolParam
from datetime import datetime


class MathAddition(ToolBase):
    def __init__(self):
        params = (
            ToolParam("first_number", "first number to add", float),
            ToolParam("second_number", "second number to add", float),
        )
        super().__init__("math_add", "Mathematically adds 2 given numbers = a+b.", params)

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
            "Performs mathematical multiplication of 2 given numbers = a*b.",
            params,
        )

    def _exec(self, first_number: float, second_number: float):
        return first_number * second_number


class GetCurrentDatetime(ToolBase):
    def __init__(self):
        super().__init__(
            "get_current_datetime",
            "Returns current date and time.",
        )

    def _exec(self):
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
