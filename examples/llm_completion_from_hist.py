from matejchain.llm import LLM
from matejchain.chat_history import ChatHistory
from matejchain.message import SysMsg, UsrMsg

TEMPERATURE = 3e-11
SEED = 1337

llm = LLM(model="gpt-4o-mini")
msgs = [
    SysMsg("You are funny chatbot, always end sentence with a joke. Be very brief."),
    UsrMsg("How to explain cars are not alive to visiting alien?"),
]
hist = ChatHistory.from_msgs(msgs, limit=5)
print(f"Hist before generation:\n{hist}")

response = llm.generate(hist, choices=3, temperature=TEMPERATURE, seed=SEED)
hist.add_many(response)

print(f"Hist after generation:\n{hist}")
