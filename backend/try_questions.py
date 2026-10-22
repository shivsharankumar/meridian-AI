from rag.retrieval import ask_question,build_retriever
import time
QUESTION=[
    "What was the total revenue in FY2024?",
    "What payment terms do servo motor suppliers get?",
    "What is the customs duty on programmable controllers imported from japan?",
    "Over how many years is filling line equipment depreciated?",
    "Who won the football world cup in 2022?"

]
for number,Question in enumerate(QUESTION,1):
    print(f"\n {number},{Question}")
    print("  "+ask_question(Question).replace("\n","\n  "))


