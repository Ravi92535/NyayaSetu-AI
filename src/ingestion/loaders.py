from pathlib import Path
from typing import List, Union

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

from ..utils import logger


class CustomDocxLoader(BaseLoader):
    """
    Table-aware Document Loader for Microsoft Word (.docx / .doc) documents.
    """

    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> List[Document]:
        try:
            import docx  # python-docx
        except ImportError:
            raise ImportError(
                "python-docx is required for DOCX parsing. "
                "Install it with `pip install python-docx`."
            )

        doc = docx.Document(self.file_path)
        paragraphs = [p.text for p in doc.paragraphs if p.text and p.text.strip()]

        table_lines = []
        for table in doc.tables:
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells if c.text and c.text.strip()]
                if cells:
                    table_lines.append(" | ".join(cells))

        full_text = "\n".join(paragraphs + table_lines)
        filename = Path(self.file_path).name

        return [
            Document(
                page_content=full_text,
                metadata={"source": self.file_path, "filename": filename, "page": 0},
            )
        ]


class MultiFormatLoader:
    """
    Unified multi-format document loader supporting PDF, DOCX, and TXT/Markdown.
    """

    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt", ".md"}

    @classmethod
    def detect_format(cls, file_path: Union[str, Path]) -> str:
        return Path(file_path).suffix.lower()

    @classmethod
    def load_file(cls, file_path: Union[str, Path]) -> List[Document]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = path.suffix.lower()
        logger.info(f"Loading document: {path.name} (format: {ext})")

        if ext == ".pdf":
            try:
                from langchain_community.document_loaders import PyPDFLoader
                docs = PyPDFLoader(str(path)).load()
            except Exception as e:
                logger.warning(f"PyPDFLoader encountered an issue: {e}. Falling back to pypdf direct read.")
                import pypdf
                reader = pypdf.PdfReader(str(path))
                docs = [
                    Document(
                        page_content=page.extract_text() or "",
                        metadata={"source": str(path), "filename": path.name, "page": idx},
                    )
                    for idx, page in enumerate(reader.pages)
                ]

        elif ext in {".docx", ".doc"}:
            docs = CustomDocxLoader(str(path)).load()

        elif ext in {".txt", ".md"}:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            docs = [
                Document(
                    page_content=content,
                    metadata={"source": str(path), "filename": path.name, "page": 0},
                )
            ]

        else:
            raise ValueError(
                f"Unsupported format '{ext}' for file {path.name}. "
                f"Supported formats: {cls.SUPPORTED_EXTENSIONS}"
            )

        # Standardize metadata across all pages
        file_size = path.stat().st_size
        for doc in docs:
            doc.metadata.setdefault("filename", path.name)
            doc.metadata.setdefault("source", str(path))
            doc.metadata.setdefault("extension", ext)
            doc.metadata.setdefault("file_size", file_size)

        logger.info(f"Successfully loaded {len(docs)} pages/sections from {path.name}")
        return docs

    @classmethod
    def load_directory(
        cls,
        dir_path: Union[str, Path],
        recursive: bool = True,
    ) -> List[Document]:
        folder = Path(dir_path)
        if not folder.exists() or not folder.is_dir():
            raise NotADirectoryError(f"Directory not found: {dir_path}")

        pattern = "**/*" if recursive else "*"
        documents: List[Document] = []

        for p in folder.glob(pattern):
            if p.is_file() and p.suffix.lower() in cls.SUPPORTED_EXTENSIONS:
                try:
                    documents.extend(cls.load_file(p))
                except Exception as e:
                    logger.error(f"Error loading {p.name}: {e}")

        logger.info(f"Loaded total of {len(documents)} document pages from {folder}")
        return documents


__all__ = ["MultiFormatLoader", "CustomDocxLoader"]

