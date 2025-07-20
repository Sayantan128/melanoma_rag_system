import numpy as np
import faiss
import pytest
import yaml
from src.knowledge_agent.vectorization import save_index_and_map
from src.knowledge_agent.ingestion     import Document
from src.knowledge_agent.agent         import KnowledgeAgent

@pytest.fixture(autouse=True)
def stub_dependencies(monkeypatch):
    # 1) Dummy CrossEncoder: shorter texts rank higher
    class DummyCE:
        def __init__(self, *args, **kwargs): pass
        def predict(self, pairs):
            # avoid zero‐division by adding 1
            return np.array([1.0 / (len(text) + 1) for _, text in pairs], dtype="float32")
    monkeypatch.setattr("src.knowledge_agent.agent.CrossEncoder", DummyCE)

    # 2) Dummy EmbeddingModel: returns zero-vectors of dimension 8
    class DummyEM:
        def __init__(self, model_name): pass
        def encode(self, texts):
            return np.zeros((len(texts), 8), dtype="float32")
    monkeypatch.setattr("src.knowledge_agent.agent.EmbeddingModel", DummyEM)

@pytest.fixture
def tmp_agent(tmp_path):
    # 0) Create data dirs for Config validation
    (tmp_path / "data" / "raw").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "processed").mkdir(parents=True, exist_ok=True)

    # 1) Build FAISS index + map
    dim = 8
    idx = faiss.IndexFlatL2(dim)
    vecs = np.eye(dim, dtype="float32")
    idx.add(vecs)
    docs = {
        i: Document(content="x" * i, metadata={"page_number": i + 1})
        for i in range(dim)
    }
    ip = tmp_path / "indices" / "faiss_index.bin"
    mp = tmp_path / "indices" / "doc_map.pkl"
    save_index_and_map(idx, docs, ip, mp)

    # 2) Write config.yaml
    cfg = {
        "data_raw_dir":    str(tmp_path / "data" / "raw"),
        "processed_dir":   str(tmp_path / "data" / "processed"),
        "index_path":      str(ip),
        "doc_map_path":    str(mp),
        "embedding_model": "ignored",
        "embedding_dim":    dim,
        "faiss_index_type":"IndexFlatL2",
        "chunking_strategy":"recursive",
        "chunk_size":      10,
        "chunk_overlap":    0,
        "semantic_threshold": 90.0,
        "cross_encoder_model":"ignored",
        "rerank_top_n":    5
    }
    cf = tmp_path / "config.yaml"
    cf.write_text(yaml.dump(cfg))

    return KnowledgeAgent(cf)

def test_faiss_only(tmp_agent):
    # FAISS-only mode: returns pages 1,2,3 for k=3
    res = tmp_agent.search("q", k=3, rerank=False)
    assert [d.metadata["page_number"] for d, _ in res] == [1, 2, 3]

def test_two_stage(tmp_agent):
    # Two-stage mode: DummyCE still yields pages 1,2,3
    res = tmp_agent.search("q", k=3, rerank=True)
    assert [d.metadata["page_number"] for d, _ in res] == [1, 2, 3]
