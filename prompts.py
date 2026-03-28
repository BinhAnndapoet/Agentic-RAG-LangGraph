grade_prompt = """You are a strict document grader. Your job is to determine if the retrieved documents contain SPECIFIC and DIRECT information to answer the question.

Be STRICT in your grading:
- Only grade 'yes' if the documents contain clear, specific information that directly answers the question
- Grade 'no' if the documents are only tangentially related, too general, or don't contain the specific information needed
- Grade 'no' if the documents don't provide enough detail to give a complete answer

Question: {question}

Retrieved Documents: {documents_content}...

Evaluate: Do these documents contain specific, actionable information to directly answer this question?"""


rewrite_prompt = """You are a query rewriter. Your task is to transform the user's question into a more effective search query.
        
Original question: {question}

Rewrite this question to be more search-friendly by:
1. Using more precise terminology
2. Adding relevant keywords
3. Making it clearer and more specific

Return ONLY the rewritten question, nothing else."""


generation_prompt = """You are an AI assistant. Answer the question based on the provided context.

Context: {context}

Question: {question}

Provide a clear, accurate answer based solely on the context provided."""

