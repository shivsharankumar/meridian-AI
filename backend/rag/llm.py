from functools import lru_cache
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from config.settings import settings
from exceptions import ConfigurationError

@lru_cache(maxsize=1) #least recently used cache, only one instance of the model is created and reused
def get_llm() -> ChatGoogleGenerativeAI | ChatOpenAI:
    """Create the chat model on first use (not at import time) and reuse it afterwards."""
    if not settings.is_cloud:
        if not settings.OPENROUTER_API_KEY:
            raise ConfigurationError(
                "APP_MODE=local needs OPENROUTER_API_KEY "
                "(GOOGLE_API_KEY is no longer used for local mode)."
            )
        return ChatOpenAI(
            model=settings.openrouter_model_name,
            temperature=settings.llm_temperature,
            api_key=settings.OPENROUTER_API_KEY,
            base_url=settings.openrouter_base_url,
        )

    if not settings.GCP_PROJECT:
        raise ConfigurationError("APP_MODE=cloud needs GCP_PROJECT_ID for Vertex AI Gemini")

    # Cloud: authenticates with the Cloud Run service account (no API key to store or rotate)
    return ChatGoogleGenerativeAI(
        model=settings.llm_model_name,
        temperature=settings.llm_temperature,
        vertexai=True,
        project=settings.GCP_PROJECT,
        location=settings.llm_location,
    )
"""`lru_cache` is a Python decorator that remembers the result of a function call. 
If you call the function again with the same inputs, Python returns the remembered result instead of running the function again.

“LRU” means **Least Recently Used**. The cache has a size limit; when it’s full, 
Python removes the result that hasn’t been used for the longest time.

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def get_answer(question):
    print("Doing the work...")
    return question.upper()

print(get_answer("hello"))  # Prints "Doing the work...", then HELLO
print(get_answer("hello"))  # Just returns HELLO; no work this time
```
It’s useful when a function is slow and is often called again with the same inputs. 
For example, it can avoid repeating an expensive calculation or fetching the same data multiple times.

One thing to keep in mind: the inputs must be hashable 
(for example, strings and numbers work; lists don’t), and cached results use memory."""

def extract_text(content)-> str:
    """Gemini may return a list of content blocks .eg.[{'type': 'text', 'text': 'Hello, World!'}]; join the text 
    Extract text from a string or a list of strings."""
    if isinstance(content, list):
        return "\n".join(block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text")
    return str(content)