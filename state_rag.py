"""
State Definitions and Pydantic Schemas for Agentic RAG

This module defines the core state objects and structured output schemas
used in the Agentic Retrieval-Augmented Generation (RAG) workflow.
It includes the RAG agent's runtime state for message tracking and retries,
as well as structured schemas to enforce consistent LLM outputs for document
grading and query rewriting.
"""

from typing_extensions import Annotated, TypedDict, Sequence
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class RAGInputState(TypedDict):
    """
    Input State for initiating the RAG process.
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]


class RAGState(RAGInputState):
    """
    Internal State for the RAG process.
    """
    retries: int


class GradeDocuments(BaseModel):
    """
    Schema for grading documents in the RAG workflow.

    Forces LLMs to return structured output when evaluating document relevance.
    - binary_score: 'yes' if documents are relevant to the query, 'no' otherwise.
    - reasoning: concise explanation of why the documents are or aren't relevant.
    """

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"        
    )

    reasoning: str = Field(
        description="Brief explanation of why documents are or aren't relevant"
    )


class RewrittenQuery(BaseModel):
    """
    Schema for optimized query rewriting in the RAG workflow.

    Forces LLMs to return only the rewritten, optimized search query to improve
    retrieval effectiveness.
    - rewritten_query: the search query optimized for retrieval.
    """

    rewritten_query: str = Field(
        description= "The optimized and rewritten search query"
    )
        
