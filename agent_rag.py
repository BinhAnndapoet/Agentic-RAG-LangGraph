from typing import Literal
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from .state_rag import RAGState, RAGInputState, GradeDocuments, RewrittenQuery
from .utils import extract_context_from_messages
from .prompts import grade_prompt, rewrite_prompt, generation_prompt
from .config.gemini import get_llm #

MAX_RETRIES = 3

model = get_llm()
structured_grader = model.with_structured_output(GradeDocuments)
structured_rewriter = model.with_structured_output(RewrittenQuery)


# ==== GRAPH NODES ====

def create_llm_call_node(tools):
    model_with_tools = model.bind_tools(tools)
    
    def llm_call(state: RAGState):
        """Analyze the request and invoke the tool if necessary."""
        response = model_with_tools.invoke(state["messages"])
        retries = state.get("retries", 0)
        return {"messages": [response], "retries": retries}
    
    return llm_call


def rewrite_query(state: RAGState):
    """Rewrite the query if the retrieved data is not satisfactory."""
    question = state["messages"][0].content
    
    formatted_prompt = rewrite_prompt.format(question=question)
    response = structured_rewriter.invoke([SystemMessage(content=formatted_prompt)])
    
    current_retries = state.get("retries", 0) + 1
    print(f"\n Optimizing the query (Attempt {current_retries}): {response.rewritten_query}")
    
    return {
        "messages": [HumanMessage(content=response.rewritten_query)], 
        "retries": current_retries
    }


def generate_final_answer(state: RAGState):
    """Compile the context and generate the final answer."""
    question = state["messages"][0].content
    context = extract_context_from_messages(state.get("messages", []))
    
    formatted_prompt = generation_prompt.format(context=context, question=question)
    response = model.invoke([SystemMessage(content=formatted_prompt)])
    
    return {"messages": [response]}


# ==== GRAPH EDGES ====

def route_after_agent(state: RAGState) -> Literal["retrieve", "__end__"]:
    """
    Decides whether to call the retrieval tool or terminate the process.

    This edge logic checks the last message to see if the agent has called a tool. If so, 
    it routes to the retrieval node; otherwise, it ends the process.
    
    Args:
        state (RAGState): The current state of the RAG process.
    
    Returns:
        Literal["retrieve", "__end__"]: The next node to route to.
    """

    messages = state["messages"]
    last_message = messages[-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "retrieve"
    return "__end__"


def grade_documents(state: RAGState) -> Literal["generate", "rewrite"]:
    """
    Grades the retrieved documents and decides whether to generate or rewrite the answer.

    This edge logic checks if the retries limit has been reached, and if so, generates
    the final answer. Otherwise, it uses the grading model to evaluate the relevance of 
    the documents and decides whether to rewrite the query or proceed with generating the 
    final answer.

    Args:
        state (RAGState): The current state of the RAG process.

    Returns:
        Literal["generate", "rewrite"]: The next step after grading.
    """

    retries = state.get("retries", 0)
    if retries >= MAX_RETRIES:
        print(f"\n Reached search limit ({MAX_RETRIES} attempts). Generating the best possible answer.")
        return "generate"
        
    messages = state["messages"]
    question = messages[0].content
    context = extract_context_from_messages(messages)
    
    if not context:
        return "generate"
    
    formatted_prompt = grade_prompt.format(question=question, documents_content=context[:1500])
    grade = structured_grader.invoke([SystemMessage(content=formatted_prompt)])
    
    print(f"\n Grading documents: {grade.binary_score.upper()}")
    print(f"Reasoning: {grade.reasoning}\n")
    
    if grade.binary_score.lower() == "yes":
        return "generate"
    else:
        return "rewrite"


# ==== GRAPH BUILDER ====

def compile_rag_graph(tools):
    """
    Initializes the RAG graph. Call this function from the main.py file.

    This function sets up the workflow for the research agent by defining nodes and edges.
    It links the model's call, the retrieval process, the rewriting process, and the 
    final answer generation. It also defines routing logic between the nodes based on 
    certain conditions.

    Args:
        tools: A list of tools to be used by the agent.

    Returns:
        StateGraph: The compiled RAG graph ready for execution.
    """

    workflow = StateGraph(RAGState, input_schema=RAGInputState)
    
    retrieve_node = ToolNode(tools)
    llm_call_node = create_llm_call_node(tools)
    
    workflow.add_node("agent", llm_call_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("rewrite", rewrite_query)
    workflow.add_node("generate", generate_final_answer)

    workflow.add_edge(START, "agent")
    
    workflow.add_conditional_edges(
        "agent",
        route_after_agent,
        {"retrieve": "retrieve", "__end__": END}
    )
    
    workflow.add_conditional_edges(
        "retrieve",
        grade_documents,
        {"generate": "generate", "rewrite": "rewrite"}
    )
    
    workflow.add_edge("rewrite", "agent")
    workflow.add_edge("generate", END)
    
    return workflow.compile()