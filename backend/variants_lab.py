from pydantic import BaseModel, Field
from rag.vector_store import get_vector_store
from rag.llm import get_llm
from langchain_core.documents import Document
from variants_data import SENTENCES
from langchain_chroma import Chroma 
from rag.embeddings import get_embeddings_model
from config.settings import settings
store=get_vector_store()
class Scores(BaseModel):
    scores:list[int]=Field(description="one relevance score from 0 to 10 per sentence, in the same order")

def rerank(question,k=3):
    """Fetch 8 candidates, let the model score each one, keep the best k"""
    candidates= store.similarity_search(question,k=8)
    numbered="\n".join(f"{n}.{d.page_content}" for n,d in enumerate(candidates,1))
    result=get_llm().with_structured_output(Scores).invoke(
        f"Question:{question}\nScore each sentence from 0 to 10 for how well it answers the question.\n{numbered}"
    )
    ranked=sorted(zip(result.scores,candidates),key=lambda pair:pair[0],reverse=True)
    return [doc for _,doc in ranked[:k]]

def parent_page(question,k=3):
    """Search small pieces, hand over the whole page they came from"""
    pages={}
    for hit in store.similarity_search(question,k=k):
        source=hit.meatdata["source"]
        text=" ".join(d.page_content for d in DOCS if d.metadata["source"]==source)
        pages[source]=Document(page_content=text,metadata={"id":source})
    return list(pages.values())
DOCS=[Document(page_content=text,metadata={"id":id_,"source":source}) for id_,source,text in SENTENCES]
By_ID={doc.metadata["id"]: doc for doc in DOCS}
store=Chroma(collection_name="variant_lab",embedding_function=get_embeddings_model(),persist_directory=settings.chroma_dir)
store.reset_collection()
store.add_documents(DOCS,ids=list(By_ID))
def basic(question,k=3):
    return store.similarity_search(question,k=k)