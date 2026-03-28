from typing import List
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.tools import create_retriever_tool
from langchain_core.vectorstores import VectorStore
from .config.gemini import get_embeddings

def ingest_documents(urls: List[str]) -> VectorStore:
    """
    Load content from the provided URLs, split it into chunks, and store it in a FAISS Vector Store.
    
    Args:
        urls: A list of URLs to read from.
        
    Returns:
        A FAISS VectorStore object containing the embedded data.
    """

    print("Downloading documents from URLs...")
    loader = WebBaseLoader(urls)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200
    )

    splits = text_splitter.split_documents(docs)

    print(f"Split into {len(splits)} text chunks.")

    embeddings = get_embeddings()

    vectorstore = FAISS.from_documents(
        documents=splits,
        embedding=embeddings
    )

    return vectorstore


def get_retrieve_tool(vectorstore: VectorStore):
    """
    Convert the VectorStore into a tool that the Agent can call.
    
    Args:
        vectorstore: The vector database (FAISS) containing the ingested data.
        
    Returns:
        A LangChain Tool object.
    """

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    retriever_tool = create_retriever_tool(
        retriever=retriever,
        name="retrieve_documents",
        description="Search and retrieve relevant documents from the knowledge base. "
                    "Use this tool when you need external information or facts to answer the question."
    )

    return retriever_tool
