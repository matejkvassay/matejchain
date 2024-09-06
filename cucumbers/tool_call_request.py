import logging
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ToolCallRequest(BaseModel):
    id: str
    name: str
    kwargs: dict
