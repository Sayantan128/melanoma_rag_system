from pathlib import Path
from typing import List
from src.knowledge_agent.ingestion import Document

# Expose these so tests can monkeypatch them
class SemanticChunker:
    def __init__(self, embeddings, breakpoint_threshold_type, breakpoint_threshold_amount):
        self.embeddings = embeddings
        self.breakpoint_threshold_type = breakpoint_threshold_type
        self.breakpoint_threshold_amount = breakpoint_threshold_amount

    def split_text(self, text: str) -> List[str]:
        # real implementation would use semantic breaks,
        # but tests will stub this out.
        return [text]


class OpenAIEmbeddings:
    def __init__(self, model_name: str):
        # real impl would initialise HF/OpenAI embedding client
        pass


def chunk_recursively(
    docs: List[Document],
    chunk_size: int,
    chunk_overlap: int
) -> List[Document]:
    """
    Naive sliding‐window splitter over the raw text.
    """
    out: List[Document] = []
    for doc in docs:
        text = doc.content
        length = len(text)
        start = 0
        while start < length:
            end = min(start + chunk_size, length)
            piece = text[start:end]
            md = dict(doc.metadata)
            md["chunk_start"] = start
            md["chunk_end"] = end
            out.append(Document(content=piece, metadata=md))
            if end == length:
                break
            start = end - chunk_overlap
    return out


def chunk_semantically(
    docs: List[Document],
    model_name: str,
    threshold: float
) -> List[Document]:
    """
    Delegates to a SemanticChunker under the hood.
    """
    embedder = OpenAIEmbeddings(model_name)
    splitter = SemanticChunker(embedder, "distance", threshold)

    out: List[Document] = []
    for doc in docs:
        pieces = splitter.split_text(doc.content)
        for idx, piece in enumerate(pieces):
            md = dict(doc.metadata)
            md["chunk_index"] = idx
            out.append(Document(content=piece, metadata=md))
    return out
