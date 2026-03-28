from langchain_core.messages import ToolMessage

def extract_context_from_messages(messages: list) -> str:
    """
    Extract document content from the message history.
    Iterate in reverse to always get the most recent retrieval result.
    """

    context = ""

    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            context = msg.content
            break

    return context

