from rag.retrieval import ask_question,build_retriever
import time
QUESTION="What payment terms do servo motor suppliers get?"
for mode in ("similarity","multiquery","contextual"):
    chunks=build_retriever(mode).invoke(QUESTION)
    print(f"\n==={mode} :{len(chunks)} chunks go into the prompt")
    for chunk in chunks:
        print(chunk)
        source = chunk.metadata.get("source", "Unknown source")
        print(f"{source} ({len(chunk.page_content)} characters)")
    started=time.time()
    answer=ask_question(QUESTION,mode)
    print(f"one full answer took {time.time() - started:.1f} s")
    print(" "+answer.replace("\n","\n "))
