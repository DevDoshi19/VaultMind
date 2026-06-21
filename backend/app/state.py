from typing import TypedDict

"""
Retriver vs Generator (RAG) is a framework that combines retrieval-based and generation-based approaches to answer questions.

- The RAG process typically involves the following steps:
1. **Question Understanding**: The system takes a user's question as input and processes it to understand the intent and key components.
2. **Document Retrieval**: The system retrieves relevant documents or information from a knowledge base or
external sources based on the processed question.
3. **Answer Generation**: The system generates an answer by synthesizing the retrieved information and the original question, often using natural language generation techniques.

"""

class RAGState(TypedDict):
    """State of the RAG process."""
    question: str              # The user's question
    question_is_relevant: bool  # Whether the question is relevant to the context
    retrieved_docs: list[str]  # List of retrieved documents relevant to the question
    retrieval_status : str
    context_token_count: int   # how many tokens ended up in context
    answer: str                # The generated answer based on the retrieved documents and question    

    confidence_score: float | None      # Confidence score of the generated answer

    prompt_tokens :int
    completion_tokens : int
    total_tokens :int
    estimated_cost :float

    input_blocked:bool           # ← confirm this exists
    output_flagged:bool