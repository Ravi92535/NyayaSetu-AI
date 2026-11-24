import re
from typing import List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..config import settings
from ..utils import logger


class DocumentChunker:
    """
    Splits documents into overlapping chunks using RecursiveCharacterTextSplitter,
    preserving full document provenance and injecting chunk identification.
    Also provides statute-aware section extraction for legal corpora.
    """

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            keep_separator=True,
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Splits a list of LangChain documents into chunks with unique IDs.
        """
        if not documents:
            return []

        logger.info(
            f"Chunking {len(documents)} document sections "
            f"(chunk_size={self.chunk_size}, overlap={self.chunk_overlap})"
        )

        chunks = self._splitter.split_documents(documents)

        for idx, chunk in enumerate(chunks):
            filename = chunk.metadata.get("filename", "unknown")
            page = chunk.metadata.get("page", 0)
            chunk.metadata["chunk_id"] = f"{filename}_p{page}_c{idx}"
            chunk.metadata["chunk_index"] = idx

        logger.info(f"Generated {len(chunks)} chunks.")
        return chunks

    def split_text(
        self,
        text: str,
        metadata: Optional[dict] = None,
    ) -> List[Document]:
        """
        Splits a raw text string directly into chunks.
        """
        meta = metadata or {}
        doc = Document(page_content=text, metadata=meta)
        return self.split_documents([doc])

    @staticmethod
    def extract_section_title(body: str, text_before: str) -> str:
        """Extract marginal section title from statutory layout."""
        blocks = re.split(r'\n\s*\n', body)
        for b in blocks:
            lines = [l.strip() for l in b.split('\n') if l.strip()]
            lines = [l for l in lines if not re.match(r'^\d+$', l) and not l.startswith('CHAPTER') and not l.startswith('PART')]
            if not lines:
                continue
            cand = ' '.join(lines)
            if (4 <= len(cand) <= 150 and cand[0].isupper() and
                not cand.startswith('(') and
                not any(w in cand.lower() for w in ['shall', 'sub-section', 'proviso', 'herein', 'provided that'])):
                return cand.rstrip('.')

        before_lines = [l.strip() for l in text_before.split('\n') if l.strip()]
        cand_lines = []
        for l in reversed(before_lines[-8:]):
            if any(l.startswith(x) for x in ['CHAPTER', 'PART', '_____', 'No.', 'THE ']) or re.match(r'^\d+$', l):
                break
            cand_lines.insert(0, l)
            cand = ' '.join(cand_lines)
            if (4 <= len(cand) <= 150 and cand[0].isupper() and
                not cand.startswith('(') and
                not any(w in cand.lower() for w in ['shall', 'sub-section', 'proviso', 'herein', 'provided that'])):
                return cand.rstrip('.')
        return ''

    def split_statute_sections(self, documents: List[Document]) -> List[Document]:
        """
        Extracts legal statute sections into distinct Document objects with section_no, title, and act metadata.
        Falls back to standard recursive chunking if no statute pattern is matched.
        """
        pattern = re.compile(r'\n([0-9]{1,3})\.\s+(.*?)(?=\n[0-9]{1,3}\.\s+|\Z)', re.DOTALL)
        section_docs: List[Document] = []

        for doc in documents:
            filename = doc.metadata.get("filename", "")
            fname = filename.lower()
            act_code = (
                "bnss" if ("nagarik" in fname or "bnss" in fname) else (
                    "bsa" if ("sakshya" in fname or "bsa" in fname) else (
                        "bns" if ("nyaya" in fname or "bns" in fname) else "act"
                    )
                )
            )
            raw_text = doc.page_content

            for m in pattern.finditer(raw_text):
                sec_num = m.group(1)
                raw_body = m.group(2).strip()
                lines = [l.strip() for l in raw_body.split('\n') if l.strip()]
                body = '\n'.join(lines)
                if len(body) < 40:
                    continue
                before = raw_text[max(0, m.start() - 400): m.start()]
                title = self.extract_section_title(body, before)
                full_text = f"{sec_num}. {body}"

                meta = dict(doc.metadata)
                meta.update({
                    "section_no": sec_num,
                    "title": title,
                    "act_code": act_code,
                    "chunk_id": f"{act_code}_sec_{sec_num}",
                })
                section_docs.append(Document(page_content=full_text, metadata=meta))

        if section_docs:
            logger.info(f"Extracted {len(section_docs)} statutory sections.")
            return section_docs
        return self.split_documents(documents)


__all__ = ["DocumentChunker"]
