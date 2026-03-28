from langchain_core.messages import ToolMessage

def extract_context_from_messages(messages: list) -> str:
    """
    Extract document content from the message history.
    Iterate in reverse to always get the most recent retrieval result.
    """

    context = ""

    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            if msg.name == "retrieve_documents":
                context = msg.content
                break

    if not context:
        print("--- WARNING: No content found from the 'retrieve_documents' tool ---")

    return context

