from matejchain.message import Message, SystemMessage
from matejchain.llm_response import LLMResponse


class ChatHistory:
    def __init__(self, limit: int = 3, sys_msg: SystemMessage | str | None = None):
        if isinstance(sys_msg, str):
            sys_msg = SystemMessage(sys_msg)
        self.sys_msg = sys_msg
        self.limit = limit
        self.messages = list()

    @classmethod
    def from_msgs(cls, messages, limit: int = 3, register_sys_msg=True):
        sys_msg = None
        if isinstance(messages[0], SystemMessage) and register_sys_msg is True:
            sys_msg = messages[0]
            messages = messages[1:]
        instance = cls(limit=limit, sys_msg=sys_msg)
        instance.add_many(messages)
        return instance

    def to_list(self) -> list[Message]:
        if self.sys_msg is None:
            return self.messages
        return [self.sys_msg] + self.messages

    def to_openai(self) -> list[dict]:
        return [m.to_openai() for m in self.to_list()]

    def add(self, message: Message | LLMResponse, append_to: list | None = None):
        if isinstance(message, LLMResponse):
            message = message.message
        self.messages.append(message)
        if append_to is not None:
            append_to.append(message)

    def add_many(self, messages: list[Message | LLMResponse], append_to: list | None = None):
        for m in messages:
            self.add(m, append_to=append_to)

    def _apply_limit(self):
        n_msgs = len(self.messages)
        if self.sys_msg is not None:
            n_msgs += 1
        if n_msgs > self.limit:
            self.messages = self.messages[1:]

    def __iter__(self):
        return self.to_list().__iter__()

    def __getitem__(self, item):
        all_msgs = self.messages
        if self.sys_msg is None:
            all_msgs = [self.sys_msg] + all_msgs
        return all_msgs.__getitem__(item)

    def __len__(self):
        if self.sys_msg is None:
            return len(self.messages)
        return len(self.messages) + 1

    def __add__(self, other):
        return self.add(other)

    def __repr__(self):
        return str(self)

    def __str__(self):
        return "\n".join(f"{i + 1}. {m}" for (i, m) in enumerate(self.to_list()))
