import numpy as np
import faiss
import pickle
import pytest
import tempfile
from pathlib import Path

# --- monkeypatch SentenceTransformer before any imports that use it ---
@pytest.fixture(autouse=True)
def _stub_sentence_transformer(monkeypatch):
    class DummyST:
        def __init__(self, model_name):
            # don’t load anything heavy
            pass
        def get_sentence_embedding_dimension(self):
            # choose a small, fixed dimension
            return 4
        def encode(self, texts, show_progress_bar=False):
            # return zero-vectors of shape (len(texts), dim)
            return np.zeros((len(texts), 4), dtype="float32")
    monkeypatch.setattr(
        "src.knowledge_agent.vectorization.SentenceTransformer",
        DummyST
    )
# --- end stub ---

from src.knowledge_agent.vectorization import (
    EmbeddingModel, save_index_and_map, load_index_and_map
)
from src.knowledge_agent.ingestion import Document

def test_embedding_model_shape():
    model = EmbeddingModel("any-model-name-is-ignored")
    texts = ["hello", "world"]
    embs = model.encode(texts)
    assert embs.shape == (2, model.get_embedding_dimension())

def test_save_and_load_index(tmp_path):
    dims = 4
    index = faiss.IndexFlatL2(dims)
    vecs = np.random.rand(3, dims).astype("float32")
    index.add(vecs)

    doc_map = {
        0: Document(content="A", metadata={"page_number":1}),
        1: Document(content="B", metadata={"page_number":2}),
        2: Document(content="C", metadata={"page_number":3}),
    }
    idx_path = tmp_path / "idx" / "test.index"
    map_path = tmp_path / "idx" / "map.pkl"

    save_index_and_map(index, doc_map, idx_path, map_path)
    loaded_index, loaded_map = load_index_and_map(idx_path, map_path)

    assert loaded_index.ntotal == index.ntotal
    assert isinstance(loaded_map, dict)
    assert loaded_map[1].content == "B"
    assert idx_path.exists() and map_path.exists()

def test_load_missing_raises(tmp_path):
    missing_idx = tmp_path / "nope.index"
    missing_map = tmp_path / "nope.pkl"
    with pytest.raises(FileNotFoundError):
        load_index_and_map(missing_idx, missing_map)
