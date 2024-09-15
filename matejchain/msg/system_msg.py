from typing import Literal

from matejchain.base import MsgBase


class SystemMsg(MsgBase):
    content: str
    role: str = Literal["system"]
