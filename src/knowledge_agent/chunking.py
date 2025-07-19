from typing import List, Any
from src.knowledge_agent.ingestion import Document

def chunk_recursively(
    documents: List[Document],
    chunk_size: int,
    chunk_overlap: int
) -> List[Document]:
    """
    Splits each Document.content into fixed-size, overlapping character chunks.
    """
    chunks: List[Document] = []
    for doc in documents:
        text = doc.content
        start = 0
        idx = 0
        step = max(chunk_size - chunk_overlap, 1)
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]
            meta = doc.metadata.copy()
            meta["chunk_id"] = f"page_{meta.get('page_number', 0)}_chunk_{idx}"
            chunks.append(Document(content=chunk_text, metadata=meta))
            idx += 1
            start += step
    return chunks

def chunk_semantically(
    documents: List[Document],
    embedding_model: Any,
    threshold: float
) -> List[Document]:
    """
    Stub semantic chunker: for now, just return one chunk per document.
    embedding_model and threshold are ignored; you can replace this
    with a real implementation later.
    """
    chunks: List[Document] = []
    for doc in documents:
        meta = doc.metadata.copy()
        meta["chunk_id"] = f"page_{meta.get('page_number', 0)}_chunk_0"
        chunks.append(Document(content=doc.content, metadata=meta))
    return chunks

