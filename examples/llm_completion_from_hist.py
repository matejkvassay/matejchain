from cucumbers.llm import LLM
from cucumbers.chat_history import ChatHistory
from cucumbers.message import SystemMessage, UserMessage

TEMPERATURE = 0.3
SEED = 1337

llm = LLM(model="gpt-4o-mini")
msgs = [
    SystemMessage("You are funny chatbot, always end sentence with a joke. Be very brief."),
    UserMessage("How to explain cars are not alive to visiting alien?"),
]
hist = ChatHistory.from_msgs(msgs, limit=5)
print(f"History before generation:\n{hist}")

response = llm.generate(hist, choices=3, temperature=TEMPERATURE, seed=SEED)
hist.add_many(response)

print(f"History after generation:\n{hist}")
