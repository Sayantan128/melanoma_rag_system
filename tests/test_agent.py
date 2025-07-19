import yaml
import faiss
import numpy as np
import pytest
from pathlib import Path

# 1. Import the class we’ll stub
from src.knowledge_agent.vectorization import EmbeddingModel
from src.knowledge_agent.vectorization import save_index_and_map
from src.knowledge_agent.ingestion     import Document
from src.knowledge_agent.agent         import KnowledgeAgent

# 2. Add this autouse fixture at the top, so every test in this module
#    uses Dummy embeddings instead of hitting HF or disk.
@pytest.fixture(autouse=True)
def _stub_embedding_model(monkeypatch):
    # Prevent the real constructor from running
    monkeypatch.setattr(EmbeddingModel, "__init__", lambda self, model_name: None)
    # Force encode() to return zero‐vectors of length 4
    monkeypatch.setattr(
        EmbeddingModel,
        "encode",
        lambda self, texts, show_progress_bar=False: np.zeros((len(texts), 4), dtype="float32")
    )

@pytest.fixture
def tmp_config(tmp_path):
    # Prepare dummy directories
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    (tmp_path / "data" / "processed").mkdir()

    # Build a tiny FAISS index and doc_map
    dim = 4
    index = faiss.IndexFlatL2(dim)
    vecs = np.eye(dim, dtype="float32")
    index.add(vecs)

    doc_map = {i: Document(content=f"doc{i}", metadata={"page_number":i+1}) for i in range(dim)}
    idx_path = tmp_path / "indices" / "faiss_index.bin"
    map_path = tmp_path / "indices" / "doc_map.pkl"
    save_index_and_map(index, doc_map, idx_path, map_path)

    # Write config.yaml
    cfg = {
        "data_raw_dir":    str(raw),
        "processed_dir":   str(tmp_path / "data" / "processed"),
        "index_path":      str(idx_path),
        "doc_map_path":    str(map_path),
        "embedding_model": "ignored",
        "embedding_dim":    dim,
        "faiss_index_type":"IndexFlatL2",
        "chunking_strategy":"recursive",
        "chunk_size":      10,
        "chunk_overlap":    0,
        "semantic_threshold": 90.0,
    }
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump(cfg))
    return config_file

def test_agent_loads_and_searches(tmp_config):
    agent = KnowledgeAgent(tmp_config)
    results = agent.search("anything", k=2)
    assert len(results) == 2
    # The first result should be doc0 (since identity vectors)
    assert results[0][0].content == "doc0"
    assert isinstance(results[0][1], float)
