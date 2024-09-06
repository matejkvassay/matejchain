from cucumbers.llm import LLM
from cucumbers.message import SystemMessage, UserMessage

llm = LLM("gpt-4o-mini")
msgs = [
    SystemMessage("You are funny chatbot, always end sentence with a joke. Be very brief."),
    UserMessage("Hey! Can you tell me about bus rides?"),
]
print(msgs)
llm_response = llm.generate_one(msgs)
print(llm_response)
