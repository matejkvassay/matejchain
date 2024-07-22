import chainlit as cl
from pepechain.chat_history import ChatHistory
from pepechain.llm import LLM
from pepechain.message import UserMessage, ToolMessage
from pepechain.tool_agent import ToolAgent
from pepechain.tools import MathAddition, MathMultiplication, GetCurrentDatetime

CHAT_HIST_LIMIT = 5
SYSTEM_PROMPT = (
    "You are Yoda, the wise Jedi high council master! Always speak like Yoda and mention "
    "anecdotes from the Star Wars canon universe after you answer."
)
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
