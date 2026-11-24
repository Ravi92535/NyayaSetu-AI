import os
import time
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

from ..config import settings
from ..retrieval import VectorStoreManager
from ..utils import logger
from .prompts import get_rag_prompt_template


def _is_valid_api_key(key: Optional[str]) -> bool:
    if not key:
        return False
    k = key.strip().lower()
    return not (k.startswith("your_") or "here" in k or len(k) < 8)


def get_llm_model(
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
) -> Any:
    """
    Resolves the Chat LLM client according to configuration.
    Raises RuntimeError if no valid API key is found.
    """
    prov = (provider or settings.LLM_PROVIDER).lower()
    model = model_name or settings.LLM_MODEL
    temp = temperature if temperature is not None else settings.LLM_TEMPERATURE

    logger.info(f"Resolving LLM model for provider '{prov}', model='{model}'")

    if prov == "groq":
        api_key = settings.GROQ_API_KEY or os.environ.get("GROQ_API_KEY")
        if not _is_valid_api_key(api_key):
            raise RuntimeError("GROQ_API_KEY is missing or invalid. Please set it in your .env file.")
        from langchain_groq import ChatGroq
        return ChatGroq(model=model, temperature=temp, api_key=api_key)

    if prov == "gemini":
        api_key = settings.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY")
        if not _is_valid_api_key(api_key):
            raise RuntimeError("GEMINI_API_KEY is missing or invalid. Please set it in your .env file.")
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model, temperature=temp, google_api_key=api_key)

    if prov == "openai":
        api_key = settings.OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY")
        if not _is_valid_api_key(api_key):
            raise RuntimeError("OPENAI_API_KEY is missing or invalid. Please set it in your .env file.")
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, temperature=temp, api_key=api_key)

    raise RuntimeError(f"Unsupported LLM provider: '{prov}'. Choose from: groq, gemini, openai.")


class RAGPipeline:
    """
    End-to-end RAG orchestrator integrating retrieval and generation.
    Uses Weaviate native hybrid search (BM25 + HNSW + RRF fusion).
    """

    def __init__(
        self,
        vector_store: Optional[VectorStoreManager] = None,
        llm: Optional[Any] = None,
    ):
        self.vector_store = vector_store or VectorStoreManager()
        self.llm = llm or get_llm_model()
        self.prompt_template = get_rag_prompt_template()

    @staticmethod
    def format_context(documents: List[Document]) -> str:
        """
        Formats documents with citation labels for prompt insertion.
        Uses Section numbers and titles for legal statutes, and page numbers for PDFs.
        """
        if not documents:
            return "No relevant context found."

        parts = []
        for idx, doc in enumerate(documents, start=1):
            source = doc.metadata.get("filename") or doc.metadata.get("source", "Unknown")
            sec_no = doc.metadata.get("section_no")
            title = doc.metadata.get("title")
            act_code = doc.metadata.get("act_code", "").upper()
            page = doc.metadata.get("page", 0)

            if sec_no:
                act_name = "BNS" if "bns" in source.lower() else ("BNSS" if "bnss" in source.lower() else ("BSA" if "bsa" in source.lower() else act_code))
                cite_label = f"{act_name} Section {sec_no}" + (f": {title}" if title else "")
            elif page > 0:
                cite_label = f"{source} (Page {page})"
            else:
                cite_label = f"{source}"

            parts.append(
                f"[Source {idx}: {cite_label}]\n{doc.page_content.strip()}"
            )
        return "\n\n".join(parts)

    def query(self, question: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        """
        Executes end-to-end RAG query: retrieve -> generate.
        Retrieval uses Weaviate native hybrid search (BM25 + HNSW + RRF fusion).
        """
        start_time = time.time()
        k = top_k or settings.TOP_K

        # 1. Retrieval (Weaviate hybrid handles BM25 + vector + RRF internally)
        final_docs = self.vector_store.similarity_search(question, k=k)

        # 3. Prompt Assembly & Generation
        context_str = self.format_context(final_docs)
        formatted_prompt = self.prompt_template.format_messages(
            context=context_str,
            question=question,
        )

        raw_response = self.llm.invoke(formatted_prompt)
        answer = raw_response.content if hasattr(raw_response, "content") else str(raw_response)

        elapsed_time = round(time.time() - start_time, 3)

        sources = [
            {
                "filename": doc.metadata.get("filename", "unknown"),
                "page": doc.metadata.get("page", 0),
                "section_no": doc.metadata.get("section_no"),
                "title": doc.metadata.get("title"),
                "act_code": doc.metadata.get("act_code", ""),
                "chunk_id": doc.metadata.get("chunk_id"),
                "snippet": doc.page_content[:200] + ("..." if len(doc.page_content) > 200 else ""),
            }
            for doc in final_docs
        ]

        return {
            "question": question,
            "answer": answer.strip(),
            "sources": sources,
            "retrieved_count": len(final_docs),
            "latency_seconds": elapsed_time,
        }


__all__ = ["RAGPipeline", "get_llm_model"]

