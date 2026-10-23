from langchain.agents import create_agent
from langchain_core.tools import tool
from rag.llm import extract_text,get_llm

@tool
def approval_needed(amount_eur:float) -> str:
    """Says who must approve a purchase of this amount,in euros."""
    if amount_eur > 1_000_000:
        return "CFO and management Board"
    if amount_eur>250_000:
        return "Department Head"
    return "The Requesting Manager"

agent=create_agent(
    model=get_llm(),
    tools=[approval_needed],
    system_prompt="You are a procurement assistant. always use your tool. Do not guess."
)

result=agent.invoke({"messages":[{"role":"user","content":"Who must approve a purchase of 480,000 EUR?"}]})
for message in result["messages"]:
    calls=getattr(message,"tool_calls",None)
    print(f"{message.type:6}",calls if calls else extract_text(message.content))