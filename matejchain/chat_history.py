from matejchain.message import Message, SystemMessage


class ChatHistory:
    def __init__(self, limit: int = 3, sys_msg: SystemMessage | str | None = None):
        if isinstance(sys_msg, str):
            sys_msg = SystemMessage(sys_msg)
        self.sys_msg = sys_msg
        self.limit = limit
        self.msgs = list()

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
            return self.msgs
        return [self.sys_msg] + self.msgs

    def to_openai(self) -> list[dict]:
        return [m.to_openai() for m in self.to_list()]

    def add(self, message: Message):
        self.msgs.append(message)
        self._apply_limit()
        return self

    def add_many(self, messages: list[Message]):
        for m in messages:
            self.add(m)

    def _apply_limit(self):
        n_msgs = len(self.msgs)
        if self.sys_msg is not None:
            n_msgs += 1
        if n_msgs > self.limit:
            self.msgs = self.msgs[1:]

    def __iter__(self):
        return self.to_list().__iter__()

    def __getitem__(self, item):
        all_msgs = self.msgs
        if self.sys_msg is None:
            all_msgs = [self.sys_msg] + all_msgs
        return all_msgs.__getitem__(item)

    def __len__(self):
        if self.sys_msg is None:
            return len(self.msgs)
        return len(self.msgs) + 1

    def __add__(self, other):
        return self.add(other)

    def __repr__(self):
        return str(self)

    def __str__(self):
        return "\n".join(f"{i + 1}. {m}" for (i, m) in enumerate(self.to_list()))
