import chainlit as cl
from cucumbers.chat_history import ChatHistory
from cucumbers.llm import LLM
from cucumbers.message import UserMessage, ToolMessage
from cucumbers.tool_agent import ToolAgent
from cucumbers.tools import MathAddition, MathMultiplication, GetCurrentDatetime

CHAT_HIST_LIMIT = 5
SYSTEM_PROMPT = "Your name is Mr. Jester Funnybot, always end your answer with a joke!"
agent = None
tools = [MathAddition(), MathMultiplication(), GetCurrentDatetime()]


@cl.on_chat_start
def on_chat_start():
    global agent
    agent = ToolAgent(
        llm=LLM("gpt-4o-mini"),
        chat_history=ChatHistory(limit=CHAT_HIST_LIMIT, sys_msg=SYSTEM_PROMPT),
        tools=tools,
    )


@cl.on_message
async def main(message: cl.Message):
    user_message = UserMessage(message.content)
    llm_messages = agent.chat(user_message)
    for m in llm_messages:
        if m.content is not None:
            if isinstance(m, ToolMessage):
                await cl.Message(
                    content=str(m),
                ).send()
            else:
                await cl.Message(
                    content=m.content,
                ).send()
