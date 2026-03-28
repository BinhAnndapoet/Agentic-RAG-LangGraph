from retriever import ingest_documents, get_retriever_tool
from agent_rag import compile_rag_graph

def save_graph_image():
    print("Đang khởi tạo graph...")
    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",  
    ]
    vectorstore = ingest_documents(urls)
    retriever_tool = get_retriever_tool(vectorstore)
    tools = [retriever_tool]
    
    app = compile_rag_graph(tools)
    
    png_bytes = app.get_graph().draw_mermaid_png()
    
    output_filename = "img/rag_graph_architecture.png"
    with open(output_filename, "wb") as f:
        f.write(png_bytes)
        
    print(f"Đã lưu sơ đồ luồng RAG thành công tại: {output_filename}")

if __name__ == "__main__":
    save_graph_image()