import chainlit as cl
from matejchain.chat_history import ChatHistory
from matejchain.llm import LLM
from matejchain.chat_agent import ChatAgent

CHAT_HIST_LIMIT = 5
SYSTEM_PROMPT = "Your name is Mr. Jester Funnybot, always end your answer with a joke!"
agent = None


@cl.on_chat_start
def on_chat_start():
    global agent
    hist = ChatHistory(limit=CHAT_HIST_LIMIT, sys_msg=SYSTEM_PROMPT)
    llm = LLM("gpt-4o-mini")
    agent = ChatAgent(llm=llm, chat_history=hist)


@cl.on_message
async def main(message: cl.Message):
    response = agent.chat(message.content).content
    await cl.Message(
        content=response,
    ).send()
