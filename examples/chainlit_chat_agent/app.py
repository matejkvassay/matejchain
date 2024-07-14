import chainlit as cl
from matejchain.chat_hist import ChatHist
from matejchain.llm import LLM
from matejchain.chat_agent import ChatAgent

hist = ChatHist(
    limit=3, sys_msg="Your name is Mr. Jester Funnybot, always end your answer with a joke!"
)
llm = LLM("gpt-3.5-turbo")
agent = ChatAgent(llm=llm, chat_hist=hist)


@cl.on_message
async def main(message: cl.Message):
    # Your custom logic goes here...

    response = agent.chat(message.content).content
    # Send a response back to the user
    await cl.Message(
        content=response,
    ).send()
