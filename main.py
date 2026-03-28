import uuid
from langchain_core.messages import HumanMessage
from retriever import ingest_documents, get_retriever_tool
from agent_rag import compile_rag_graph

def main():
    print("INITIALIZING AGENTIC RAG SYSTEM...")
    print("-" * 50)
    
    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",  
    ]
    
    vectorstore = ingest_documents(urls)
    retriever_tool = get_retriever_tool(vectorstore)
    tools = [retriever_tool]
    
    app = compile_rag_graph(tools)

    thread_id = str(uuid.uuid4)
    config = {"configurable": {"thread_id": thread_id}}
    
    print(f"\n SYSTEM IS READY! (Session: {thread_id})")
    print("(Type 'q' to quit)")
    print("=" * 50)

    while True:
        user_input = input("\n User: ")
        
        if user_input.strip().lower() == 'q':
            print("Goodbye! See you next time.")
            break
            
        if not user_input.strip():
            continue

        inputs = {"messages": [HumanMessage(content=user_input)]}
        
        print("\n The Agent is working...")
        

        final_state = None
        for output in app.stream(inputs, config=config):
            for key, value in output.items():
                print(f"  Processing at Node: [{key.upper()}]")
                final_state = value

        if final_state and "messages" in final_state:
            final_message = final_state["messages"][-1]
            print("\n Answer:")
            print("-" * 50)
            print(final_message.content)
            print("-" * 50)

if __name__ == "__main__":
    main()