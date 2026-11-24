import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import weaviate
import weaviate.classes.config as wvcc
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from weaviate.classes.query import MetadataQuery

from ..config import settings
from ..utils import logger
from .embeddings import get_embedding_model


class VectorStoreManager:
    """
    Production Vector Store Manager powered natively by Weaviate.
    Leverages Weaviate's native inverted index for BM25 and HNSW vector index
    for hybrid dense+sparse ranking without requiring custom Python ranking code.
    """

    def __init__(
        self,
        collection_name: Optional[str] = None,
        persist_directory: Optional[Union[str, Path]] = None,
        embeddings: Optional[Embeddings] = None,
        alpha: Optional[float] = None,
    ):
        self.collection_name = collection_name or settings.WEAVIATE_COLLECTION
        self.persist_directory = Path(persist_directory or (settings.BASE_DIR / settings.VECTORSTORE_DIR))
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.embeddings = embeddings or get_embedding_model()
        self.alpha = alpha if alpha is not None else settings.WEAVIATE_HYBRID_ALPHA

        self.client = self._connect()
        self.collection = self._ensure_collection()

    def _connect(self) -> weaviate.WeaviateClient:
        """
        Connects to Weaviate. Uses external URL if configured, otherwise starts
        an embedded local instance persisting data to the configured directory.
        """
        if settings.WEAVIATE_URL:
            logger.info(f"Connecting to remote Weaviate at {settings.WEAVIATE_URL}...")
            return weaviate.connect_to_custom(
                http_host=settings.WEAVIATE_URL,
                http_port=8080,
                http_secure=False,
                grpc_host=settings.WEAVIATE_URL,
                grpc_port=50051,
                grpc_secure=False,
            )

        logger.info(
            f"Connecting to Weaviate Embedded (data path: '{self.persist_directory}')..."
        )
        try:
            return weaviate.connect_to_embedded(
                persistence_data_path=str(self.persist_directory)
            )
        except weaviate.exceptions.WeaviateStartUpError as e:
            if "already listening" in str(e).lower() or "already running" in str(e).lower():
                logger.info(
                    "Existing Weaviate instance detected on ports 8079/50050. Connecting to running instance..."
                )
                return weaviate.connect_to_local(port=8079, grpc_port=50050)
            raise

    def _ensure_collection(self):
        """
        Ensures the collection schema exists with native BM25 tokenization.
        """
        if self.client.collections.exists(self.collection_name):
            return self.client.collections.get(self.collection_name)

        logger.info(
            f"Creating Weaviate collection '{self.collection_name}' with native BM25 indexing..."
        )
        return self.client.collections.create(
            name=self.collection_name,
            properties=[
                wvcc.Property(
                    name="content",
                    data_type=wvcc.DataType.TEXT,
                    tokenization=wvcc.Tokenization.WORD,  # Inbuilt BM25 word tokenization
                ),
                wvcc.Property(
                    name="title",
                    data_type=wvcc.DataType.TEXT,
                    tokenization=wvcc.Tokenization.WORD,
                ),
                wvcc.Property(name="filename", data_type=wvcc.DataType.TEXT),
                wvcc.Property(name="section_no", data_type=wvcc.DataType.TEXT),
                wvcc.Property(name="act_code", data_type=wvcc.DataType.TEXT),
                wvcc.Property(name="chunk_id", data_type=wvcc.DataType.TEXT),
                wvcc.Property(name="page", data_type=wvcc.DataType.INT),
            ],
        )

    def add_documents(self, documents: List[Document]) -> int:
        """
        Batch adds documents into Weaviate with their precomputed dense vector embeddings.
        """
        if not documents:
            return 0

        logger.info(f"Indexing {len(documents)} document chunks into Weaviate...")
        texts = [
            f"{d.metadata.get('title', '')} {d.page_content}".strip()
            for d in documents
        ]
        vectors = self.embeddings.embed_documents(texts)

        with self.collection.batch.dynamic() as batch:
            for idx, (doc, vec) in enumerate(zip(documents, vectors)):
                meta = doc.metadata or {}
                title = meta.get("title", "")
                content = f"{title}\n{doc.page_content}" if title else doc.page_content

                batch.add_object(
                    properties={
                        "content": content,
                        "title": title,
                        "filename": str(meta.get("filename", "")),
                        "section_no": str(meta.get("section_no", "")),
                        "act_code": str(meta.get("act_code", "")),
                        "chunk_id": str(meta.get("chunk_id", f"chunk_{idx}")),
                        "page": int(meta.get("page", 0)),
                    },
                    vector=vec,
                )

        logger.info(f"Successfully indexed {len(documents)} chunks in Weaviate.")
        return len(documents)

    def similarity_search_with_score(
        self,
        query: str,
        k: Optional[int] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Executes native Weaviate Hybrid search (inbuilt BM25 + dense vectors).
        """
        top_k = k or settings.TOP_K
        query_vector = self.embeddings.embed_query(query) if self.alpha > 0.0 else None

        if self.alpha == 0.0:
            res = self.collection.query.bm25(
                query=query,
                limit=top_k,
                return_metadata=MetadataQuery(score=True),
            )
        else:
            res = self.collection.query.hybrid(
                query=query,
                vector=query_vector,
                alpha=self.alpha,
                limit=top_k,
                return_metadata=MetadataQuery(score=True),
            )

        results: List[Tuple[Document, float]] = []
        for obj in res.objects:
            props = obj.properties
            content = props.get("content", "")
            title = props.get("title", "")
            # If title was prefixed, strip title header for clean page_content
            page_content = content
            if title and content.startswith(f"{title}\n"):
                page_content = content[len(f"{title}\n"):]

            metadata = {
                "title": title,
                "filename": props.get("filename", ""),
                "section_no": props.get("section_no", ""),
                "act_code": props.get("act_code", ""),
                "chunk_id": props.get("chunk_id", ""),
                "page": props.get("page", 0),
            }
            score = float(obj.metadata.score) if obj.metadata and obj.metadata.score is not None else 1.0
            results.append((Document(page_content=page_content, metadata=metadata), score))

        return results

    def similarity_search(self, query: str, k: Optional[int] = None) -> List[Document]:
        """
        Executes native hybrid search and returns a list of LangChain Document objects.
        """
        return [doc for doc, _ in self.similarity_search_with_score(query, k=k)]

    def count(self) -> int:
        """
        Returns total number of indexed objects in the Weaviate collection.
        """
        try:
            return len(self.collection)
        except Exception:
            return 0

    def as_retriever(self, k: Optional[int] = None):
        """
        Returns a LangChain-compatible retriever interface.
        """
        class _WeaviateRetriever:
            def __init__(self, manager: "VectorStoreManager", top_k: int):
                self.manager = manager
                self.top_k = top_k

            def invoke(self, query: str) -> List[Document]:
                return self.manager.similarity_search(query, k=self.top_k)

        return _WeaviateRetriever(self, k or settings.TOP_K)

    def close(self):
        """
        Closes connection to Weaviate.
        """
        if self.client:
            self.client.close()


__all__ = ["VectorStoreManager"]
