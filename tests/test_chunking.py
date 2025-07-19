import pytest
from src.knowledge_agent.chunking import chunk_recursively, chunk_semantically
from src.knowledge_agent.ingestion import Document

# --- STUB OUT external dependencies for semantic chunking ---
@pytest.fixture(autouse=True)
def patch_semantic_and_embeddings(monkeypatch):
    # 1) Stub out the LangChain SemanticChunker so it doesn't call HF:
    class DummySplitter:
        def __init__(self, embeddings, breakpoint_threshold_type, breakpoint_threshold_amount):
            pass
        def split_text(self, text):
            # just split in half for test purposes
            mid = len(text) // 2
            return [text[:mid], text[mid:]]
    monkeypatch.setattr(
        "src.knowledge_agent.chunking.SemanticChunker",
        DummySplitter
    )

    # 2) Stub out OpenAIEmbeddings constructor (so no API key needed)
    monkeypatch.setattr(
        "src.knowledge_agent.chunking.OpenAIEmbeddings",
        lambda model_name: None
    )
# --- end stubs ---


@pytest.fixture
def sample_docs():
    return [
        Document(
            content="Hello world. This is a test of chunking.",
            metadata={"page_number": 1, "total_pages": 1}
        )
    ]

def test_recursive_chunks(sample_docs):
    chunks = chunk_recursively(sample_docs, chunk_size=10, chunk_overlap=2)
    # Basic sanity: you got at least 2 chunks, and each is a Document
    assert len(chunks) > 1
    assert all(isinstance(c, Document) for c in chunks)

def test_semantic_chunks(sample_docs):
    # Now signature is (docs, model_name:str, threshold:float)
    chunks = chunk_semantically(sample_docs, model_name="anything", threshold=50.0)
    # Our DummySplitter returns exactly 2 parts
    assert len(chunks) == 2
    assert all(isinstance(c, Document) for c in chunks)
    # The two chunks, concatenated, must re-form the original text
    reconstructed = "".join(c.content for c in chunks)
    assert reconstructed == sample_docs[0].content



