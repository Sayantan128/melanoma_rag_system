from typing import List
from src.knowledge_agent.ingestion import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

def chunk_recursively(
    documents: List[Document],
    chunk_size: int,
    chunk_overlap: int
) -> List[Document]:
    """
    Split each Document.content into fixed-size, overlapping character chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
        is_separator_regex=False,
    )

    chunks: List[Document] = []
    for doc in documents:
        for i, text in enumerate(splitter.split_text(doc.content)):
            meta = doc.metadata.copy()
            meta["chunk_id"] = f"page_{meta.get('page_number', 0)}_chunk_{i}"
            chunks.append(Document(content=text, metadata=meta))
    return chunks

def chunk_semantically(
    documents: List[Document],
    model_name: str,
    threshold: float
) -> List[Document]:
    """
    Split each Document.content into semantically coherent chunks.

    model_name: HF sentence-transformers identifier (e.g. "all-MiniLM-L6-v2")
    threshold: percentile for detecting topic shifts.
    """
    # Instantiate a LangChain embeddings wrapper
    embeddings = OpenAIEmbeddings(model_name=model_name)

    splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=threshold,
    )

    chunks: List[Document] = []
    for doc in documents:
        for i, text in enumerate(splitter.split_text(doc.content)):
            meta = doc.metadata.copy()
            meta["chunk_id"] = f"page_{meta.get('page_number', 0)}_chunk_{i}"
            chunks.append(Document(content=text, metadata=meta))
    return chunks
