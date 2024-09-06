from openai.types.chat import (
    ChatCompletionMessage,
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
        if isinstance(self.openai_param, ChatCompletionMessage):
            return self.openai_param.role
        return self.openai_param.get("role")

    @property
    def content(self):
        if isinstance(self.openai_param, ChatCompletionMessage):
            return self.openai_param.content
        return self.openai_param.get("content")

    def to_openai(self) -> ChatCompletionMessageParam:
        return self.openai_param

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        role = self.role.upper()
        content = self.content
        if self.role == "assistant":
            if self.openai_param.tool_calls is not None:
                tool_call_names = [t.function.name for t in self.openai_param.tool_calls]
                if content is None:
                    return f"{role}: call tools -> {tool_call_names}"
                else:
                    return f"{role}: {content}; call tools -> {tool_call_names}"
        if self.role == "tool":
            if self.kwargs:
                return f"{role}: {self.name}({self.kwargs}) -> {content}"
            else:
                return f"{role}: {self.name}() -> {content}"
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
    def __init__(self, call_id: str, name: str, kwargs: str | None, result: str):
        kv_data = {"role": "tool", "tool_call_id": call_id, "content": result}
        self.name = name
        self.kwargs = kwargs
        super().__init__(ChatCompletionToolMessageParam(**kv_data))
