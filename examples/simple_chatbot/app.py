import chainlit as cl
from matejchain.chat_history import ChatHistory
from matejchain.llm import LLM
from matejchain.message import UserMessage
from matejchain.tool_agent import ToolAgent

CHAT_HIST_LIMIT = 5
SYSTEM_PROMPT = "Your name is Mr. Jester Funnybot, always end your answer with a joke!"
agent = ToolAgent(
    llm=LLM("gpt-4o-mini"), chat_history=ChatHistory(limit=CHAT_HIST_LIMIT, sys_msg=SYSTEM_PROMPT)
)


@cl.on_chat_start
def on_chat_start():
    global agent
    agent = ToolAgent(
        llm=LLM("gpt-4o-mini"),
        chat_history=ChatHistory(limit=CHAT_HIST_LIMIT, sys_msg=SYSTEM_PROMPT),
    )


@cl.on_message
async def main(message: cl.Message):
    user_message = UserMessage(message.content)
    llm_messages = agent.chat(user_message)
    for m in llm_messages:
        await cl.Message(
            content=m.content,
        ).send()
