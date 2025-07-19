import pytest
from src.knowledge_agent.chunking import chunk_recursively, chunk_semantically
from src.knowledge_agent.ingestion import Document

@pytest.fixture
def sample_docs():
    return [Document(content="Hello world. This is a test of chunking.", metadata={"page_number":1, "total_pages":1})]

def test_recursive_chunks(sample_docs):
    chunks = chunk_recursively(sample_docs, chunk_size=10, chunk_overlap=2)
    assert all(isinstance(c, Document) for c in chunks)
    total_chars = sum(len(c.content) for c in chunks)
    assert total_chars >= len(sample_docs[0].content)

def test_semantic_chunks(sample_docs):
    # Dummy embedding: not used by stub
    class DummyEmb: pass

    chunks = chunk_semantically(sample_docs, DummyEmb(), threshold=50.0)
    assert isinstance(chunks, list)
    assert all(isinstance(c, Document) for c in chunks)
    # Since stub returns one chunk per doc:
    assert len(chunks) == len(sample_docs)



