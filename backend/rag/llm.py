from functools import lru_cache
from langchain_google_genai import ChatGoogleGenerativeAI
from config.settings import settings
from exceptions import ConfigurationError

@lru_cache(maxsize=1) #least recently used cache, only one instance of the model is created and reused
def get_llm() -> ChatGoogleGenerativeAI:
    """Create the chat model on first use (not at import time) and reuse it afterwards."""
    if not settings.GOOGLE_API_KEY:
        raise ConfigurationError("Google api key is empty — please add GOOGLE_API_KEY in your environment or .env file.")

    return ChatGoogleGenerativeAI(
        model=settings.llm_model_name,
        temperature=settings.llm_temperature,
        google_api_key=settings.GOOGLE_API_KEY
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