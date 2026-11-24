from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

RAG_SYSTEM_PROMPT = """You are an expert AI assistant specializing in precise, factual document analysis and legal/technical retrieval.

Your instructions:
1. Answer the user's question based strictly on the provided Context excerpts below.
2. If the answer cannot be determined from the Context, state clearly: "I cannot find sufficient information in the provided context to answer this question." Do not fabricate or speculate.
3. Explicitly cite your sources using the format [Filename, Page/Section] when asserting facts.
4. Keep explanations clear, well-structured, and concise.

Context:
---------------------
{context}
---------------------"""

RAG_USER_PROMPT = """Question: {question}

Answer:"""


def get_rag_prompt_template() -> ChatPromptTemplate:
    """
    Constructs a LangChain ChatPromptTemplate for the RAG pipeline.
    """
    return ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(RAG_SYSTEM_PROMPT),
        HumanMessagePromptTemplate.from_template(RAG_USER_PROMPT),
    ])


__all__ = [
    "RAG_SYSTEM_PROMPT",
    "RAG_USER_PROMPT",
    "get_rag_prompt_template",
]

