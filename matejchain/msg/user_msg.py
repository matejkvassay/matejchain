from typing import Literal

from matejchain.base import MsgBase


class UserMsg(MsgBase):
    content: str
    role: str = Literal["user"]
