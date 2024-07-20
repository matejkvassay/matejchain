from matejchain.llm import LLM
from matejchain.message import SysMsg, UsrMsg

llm = LLM("gpt-4o-mini")
msgs = [
    SysMsg("You are funny chatbot, always end sentence with a joke. Be very brief."),
    UsrMsg("Hey! Can you tell me about bus rides?"),
]
response, _ = llm.generate_one(msgs)
print(f"Prompt: {msgs}")
print(f"LLM response: {response})")
