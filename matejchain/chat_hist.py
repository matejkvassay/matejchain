from matejchain.msg import SysMsg, Msg


class ChatHist:
    """
    Records chat history containing last <limit> messages.

    Optionally system message can be passed to constructor.
    This message will then always occupy 1st place in chat history as new messages are added.
    Keep in mind setting system message will allocate 1 slot from chat history size <limit>.

    Example initializing and adding 2 messages to history in 2 alternative ways:
        ```
        hist = ChatHist(limit=5, sys_msg="You are a helpful assistant")
        msg1 = UsrMsg()
        msg2 = AssMsg()
        hist = hist + msg1
        hist.add(msg2)
        ```
    """

    def __init__(self, limit: int = 3, sys_msg: SysMsg | str | None = None):
        """
        :param limit: int maximum number of messages held in this history.
                      If limit exceeded on add(), the oldest message is removed.
                      If sys_msg is passed it takes 1 slot from history and
                      limit left for other messages = limit-1.
        :param sys_msg: optional, pass str or SysMsg object to set main System message.
                        This msg will always be kept in 1st position of the
                        history and won't be affected by limit.
                        This however will leave only limit-1 free history slots for other messages.
        """
        if isinstance(sys_msg, str):
            sys_msg = SysMsg(sys_msg)
        self.sys_msg = sys_msg
        self.limit = limit
        self.msgs = list()

    @classmethod
    def from_msgs(cls, msgs, limit: int = 3, register_sys_msg=True):
        """

        :param msgs: List of Msg instances.
        :param limit: int maximum number of messages held in this history.
                      If limit exceeded on add(), the oldest message is removed.
                      If sys_msg is passed it takes 1 slot from history and
                      limit left for other messages = limit-1.
        :param register_sys_msg: If True and there is system message on index 0 in msgs,
                                 then this system message will be set as the main history
                                 system message (see __init__ docs).
                                 If there is not a system message on index 0 then
                                 it does nothing (sys_msg = None).
                                 If False then it does nothing (sys_msg = None).
        :return: instance of ChatHist with msgs filled in.
        """
        sys_msg = None
        if isinstance(msgs[0], SysMsg) and register_sys_msg is True:
            sys_msg = msgs[0]
            msgs = msgs[1:]
        instance = cls(limit=limit, sys_msg=sys_msg)
        instance.add_many(msgs)
        return instance

    def to_list(self) -> list[Msg]:
        """
        Get message history as list of messages.

        :return: list of messages (Msg), oldest on lowest index
        """
        if self.sys_msg is None:
            return self.msgs
        return [self.sys_msg] + self.msgs

    @property
    def openai_fmt(self) -> list[dict]:
        """
        Get list of dictionaries compatible with OpenAI completions API.
        Oldest message is on lowest index.

        :return: list of dicts, can be passed to OpenAI client as messages
                 e.g. [{"role": "<?>", "content": "<?>"}, ... ]
        """
        return [x.openai_fmt for x in self.to_list()]

    def add(self, msg: Msg):
        """
        Add new message to the end of history.

        :param msg: Any message, child of Msg class
        """
        self.msgs.append(msg)
        self._apply_limit()
        return self

    def add_many(self, msgs: list[Msg]):
        """
        Add multiple new messages from given list to the end of history.

        :param msgs: list of Msg instances
        """
        for m in msgs:
            self.add(m)

    def _apply_limit(self):
        """
        Cuts message list in history to configured limit.
        Keeps sys_msg on top if sys_msg configured.
        """
        n_msgs = len(self.msgs)
        if self.sys_msg is not None:
            n_msgs += 1
        if n_msgs > self.limit:
            self.msgs = self.msgs[1:]

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
