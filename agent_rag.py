from typing import Literal
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from state_rag import RAGState, RAGInputState, GradeDocuments, RewrittenQuery
from utils import extract_context_from_messages
from prompts import grade_prompt, rewrite_prompt, generation_prompt
from config.gemini import get_llm 

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


def rewrite_query_node(state: RAGState):
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


def grade_documents_node(state: RAGState):
    """Grading and save the result into the state."""
    messages = state["messages"]
    question = messages[0].content
    context = extract_context_from_messages(messages)
    
    if not context:
        return {"retries": state.get("retries", 0), "messages": [SystemMessage(content="no_context")]}

    formatted_prompt = grade_prompt.format(question=question, documents_content=context[:1500])
    
    grade = structured_grader.invoke([HumanMessage(content=formatted_prompt)]) 
    
    return {
        "messages": [HumanMessage(content=grade.binary_score, name="grader_score")],
        "retries": state.get("retries", 0)
    }


def generate_final_answer_node(state: RAGState):
    """Compile the context and generate the final answer."""
    messages = state.get("messages", [])
    if not messages:
        return {"messages": [HumanMessage(content="I don't have enough information to answer.")]}
        
    question = messages[0].content
    context = extract_context_from_messages(messages)
    
    if not context or context.strip() == "":
        context = "No relevant documents were found to answer this question."

    formatted_prompt = generation_prompt.format(context=context, question=question)
    
    # Checking before generating answer
    print(f"CHECK - Question: {question}")
    print(f"CHECK - Context Length: {len(context)}")
    print(f"CHECK - Formatted Prompt: {formatted_prompt[:100]}...")
    
    response = model.invoke([HumanMessage(content=formatted_prompt)])
    
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


def decide_to_generate(state: RAGState) -> Literal["generate", "rewrite"]:
    """
    Decides whether to generate the final answer or rewrite the query based on the grading result and retry count.
    
    This function inspects the latest message in the state to determine if the grading result was positive 
    (i.e., contains the word "yes"). If the grading result is positive or if the retry count has reached 
    the maximum allowed retries (`MAX_RETRIES`), it returns the decision to generate the final answer.
    Otherwise, it returns the decision to rewrite the query and attempt again.

    Args:
        state (RAGState): The current state of the RAG process, including messages and retry count.
        
    Returns:
        Literal["generate", "rewrite"]: 
            - "generate" if the grading result is positive or the maximum retry limit is reached.
            - "rewrite" if the grading result is negative and retries are still within the limit.
    """

    last_message = state["messages"][-1]
    retries = state.get("retries", 0)

    if retries >= MAX_RETRIES:
        return "generate"

    if "yes" in last_message.content.lower():
        return "generate"
    return "rewrite"


# ==== GRAPH BUILDER ====

def compile_rag_graph(tools):
    workflow = StateGraph(RAGState, input_schema=RAGInputState)
    
    workflow.add_node("agent", create_llm_call_node(tools))
    workflow.add_node("retrieve", ToolNode(tools))
    workflow.add_node("grade_docs", grade_documents_node) 
    workflow.add_node("rewrite", rewrite_query_node)
    workflow.add_node("generate", generate_final_answer_node)

    workflow.add_edge(START, "agent")
    
    workflow.add_conditional_edges(
        "agent",
        route_after_agent,
        {"retrieve": "retrieve", "__end__": END}
    )

    workflow.add_edge("retrieve", "grade_docs")

    workflow.add_conditional_edges(
        "grade_docs",
        decide_to_generate,
        {"generate": "generate", "rewrite": "rewrite"}
    )

    workflow.add_edge("rewrite", "agent")
    workflow.add_edge("generate", END)

    return workflow.compile()