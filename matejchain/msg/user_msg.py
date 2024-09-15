from matejchain.base import MsgBase


class UserMsg(MsgBase):
    content: str
    role: str = "user"
