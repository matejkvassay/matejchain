from pydantic import BaseModel
from typing import Optional
from matejchain.base.api_compatible_base import ApiCompatibleBase


class MsgBase(BaseModel, ApiCompatibleBase):
    role: str
    content: Optional[str]

    def api_dict(self) -> dict:
        return self.dict()

    def __str__(self):
        content = self.content if self.content is not None else "<no content>"
        return f'{self.role.upper()}: "{content}"'

    def __repr__(self):
        return self.__str__()
