from langchain.chat_models import ChatGoogle
from langchain.embeddings import GoogleEmbeddings
from .settings import GEMINI_API_KEY 

def get_llm(model_name: str = "gemini-2.5-flash", temperature: float = 0):
    return ChatGoogle(
        model=model_name,
        temperature=temperature,
        api_key=GEMINI_API_KEY
    )

def get_embeddings():
    return GoogleEmbeddings(api_key=GEMINI_API_KEY)