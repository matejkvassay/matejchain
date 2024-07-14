from matejchain.llm import LLM
from matejchain.msg import SysMsg, UsrMsg

llm = LLM("gpt-3.5-turbo")
msgs = [
    SysMsg("You are funny chatbot, always end sentence with a joke. Be very brief."),
    UsrMsg("Hey! Can you tell me about bus rides?"),
]
response, _ = llm.generate_one(msgs)
print(f"Prompt: {msgs}")
print(f"LLM response: {response})")
