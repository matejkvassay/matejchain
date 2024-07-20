from openai.types.chat import ChatCompletionMessage
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
    ChatCompletionToolMessageParam,
)


class Message:
    def __init__(self, openai_param):
        self.openai_param = openai_param

    @property
    def role(self):
        return self.openai_param.role

    @property
    def content(self):
        return self.openai_param.content

    def to_openai(self) -> ChatCompletionMessageParam:
        return self.openai_param

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        role = self.openai_param.role.upper()
        content = self.openai_param.content
        if isinstance(self.openai_param, ChatCompletionAssistantMessageParam):
            if self.openai_param.tool_calls is not None:
                tool_call_names = [t.name for t in self.openai_param.tool_calls]
                if content is None:
                    return f"{role}: call tools -> {tool_call_names}"
                else:
                    return f"{role}: {content}; call tools -> {tool_call_names}"
        if isinstance(self.openai_param, ChatCompletionToolMessageParam):
            return f"{role}:{self.openai_param.tool_call_id} {content}"
        return f"{role}: {content}"


class SystemMessage(Message):
    def __init__(self, content: str, participant_name: str | None = None):
        kv_data = {"role": "system", "content": content}
        if participant_name is not None:
            kv_data["name"] = participant_name
        super().__init__(ChatCompletionSystemMessageParam(**kv_data))


class UserMessage(Message):
    def __init__(self, content: str, participant_name: str | None = None):
        kv_data = {"role": "user", "content": content}
        if participant_name is not None:
            kv_data["name"] = participant_name
        super().__init__(ChatCompletionUserMessageParam(**kv_data))


class AssistantMessage(Message):
    def __init__(
            self, completions_message: ChatCompletionAssistantMessageParam | ChatCompletionMessage
    ):
        super().__init__(completions_message)


class ToolMessage(Message):
    def __init__(self, tool_call_id: str, tool_call_result: str):
        kv_data = {"role": "tool", "tool_call_id": tool_call_id, "content": tool_call_result}
        super().__init__(ChatCompletionToolMessageParam(**kv_data))
