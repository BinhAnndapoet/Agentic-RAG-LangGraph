from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from .settings import GEMINI_API_KEY

def get_llm(model_name: str = "gemini-2.5-flash", temperature: float = 0):
    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        api_key=GEMINI_API_KEY,

    )

def get_embeddings(model_name: str = "models/text-embedding-004"):
    return GoogleGenerativeAIEmbeddings(
        model=model_name,
        google_api_key=GEMINI_API_KEY
    )