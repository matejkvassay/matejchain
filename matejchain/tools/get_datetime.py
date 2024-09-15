from matejchain.base import ToolBase
from datetime import datetime


class GetDatetime(ToolBase):
    def __init__(self):
        super().__init__(
            name="get_current_datetime",
            desc="Returns current date and time.",
        )

    @staticmethod
    def exec(self):
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
