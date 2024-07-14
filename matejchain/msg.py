from openai.types.chat import ChatCompletionMessage

ROLE_SYSTEM = "system"
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
ROLE_TOOL = "tool"


class Msg:
    """
    Base class for chat message.

    Use child classes UserMsg, SystemMsg, AssistantMsg or ToolMsg to create
    regular messages for chat.

    Use this class only to create new type of message with custom role/content/behaviour.

    To retrieve OpenAI format {"role":X, "content": X} use class attribute "openai_fmt"
    or method "to_openai()"
    """

    def __init__(self, role: str, content: str | ChatCompletionMessage):
        """
        :param role: OpenAI completions API compatible role, e.g. "user"
        :param content: message content str or ChatCompletionMessage
        """
        if isinstance(content, ChatCompletionMessage):
            content = content.content
        if content is None:
            raise ValueError("Cannot initialize message obj with empty content.")
        self.content = content
        self.role = role
        self.openai_fmt = {"role": self.role, "content": self.content}

    def to_openai(self) -> dict:
        """
        Get dict compatible with OpenAI completions api message format.

        :return: dict fmt {"role": <ROLE>, "content": <CONTENT>}
        """
        return self.openai_fmt

    def is_sys_msg(self) -> bool:
        """
        True if this message is System message.
        """
        return self.role == ROLE_SYSTEM

    def is_usr_msg(self):
        """
        True if this message is User message.
        """
        return self.role == ROLE_USER

    def is_ass_msg(self) -> bool:
        """
        True if this message is Assistant message.
        """
        return self.role == ROLE_ASSISTANT

    def is_tool_msg(self) -> bool:
        """
        True if this message is Tool message.
        """
        return self.role == ROLE_TOOL

    def __eq__(self, other):
        return (other.role == self.role) and (other.content == self.content)

    def __str__(self):
        return f"{self.role.upper()}: {self.content}"

    def __repr__(self):
        return self.__str__()


class SysMsg(Msg):
    """
    System message. Usually used in beginning of chat to configure persona and basic
    instructions to condition the chat LLM to alter its behaviour and tone.

    To retrieve OpenAI format {"role":X, "content": X} use class attribute "openai_fmt"
    or method "to_openai()"
    """

    def __init__(self, content: str):
        """
        :param content: Message content.
        """
        super().__init__(ROLE_SYSTEM, content)


class UsrMsg(Msg):
    """
    User message. Represents messages from chatbot user.

    To retrieve OpenAI format {"role":X, "content": X} use class attribute "openai_fmt"
    or method "to_openai()"
    """

    def __init__(self, content):
        """
        :param content: Message content.
        """
        super().__init__(ROLE_USER, content)


class AssMsg(Msg):
    """
    Assistant message. Represents messages from the LLM chatbot.

    To retrieve OpenAI format {"role":X, "content": X} use class attribute "openai_fmt"
    or method "to_openai()"
    """

    def __init__(self, content):
        """
        :param content: Message content.
        """
        super().__init__(ROLE_ASSISTANT, content)


class ToolMsg(Msg):
    """
    Represents output from tool execution.

    To retrieve OpenAI format {"role":X, "content": X} use class attribute "openai_fmt"
    or method "to_openai()"
    """

    def __init__(self, content: str):
        """
        :param content: Message content.
        """
        super().__init__(ROLE_TOOL, content)
