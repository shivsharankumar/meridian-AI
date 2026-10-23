import json
import requests
from langchain_core.tools import tool
from typing import Dict
from rag.llm import extract_text,get_llm
from logger import GLOBAL_LOGGER as log

#internal LLm Helper
def _ask_llm(prompt:str) -> str:
    """Invoke the configured LLM and return the text content."""
    try:
        response=get_llm().invoke(prompt)
        return extract_text(response.content)
    except Exception as e:
        return f"LLM Error:{str(e)}"

#step 1 : Define dynamic tool sets