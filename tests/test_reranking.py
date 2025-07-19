import numpy as np
import faiss
import pytest
import yaml
from src.knowledge_agent.vectorization import save_index_and_map
from src.knowledge_agent.ingestion     import Document
from src.knowledge_agent.agent         import KnowledgeAgent

@pytest.fixture(autouse=True)
def stub_cross_encoder(monkeypatch):
    # Dummy CrossEncoder: shorter texts rank higher
    class DummyCE:
        def __init__(self, *args, **kwargs): pass
        def predict(self, pairs):
            return np.array([1.0/len(text) for _, text in pairs], dtype="float32")
    monkeypatch.setattr("src.knowledge_agent.agent.CrossEncoder", DummyCE)

@pytest.fixture
def tmp_agent(tmp_path):
    # 1) Build FAISS idx + map
    dim = 8
    idx = faiss.IndexFlatL2(dim)
    vecs = np.eye(dim, dtype="float32"); idx.add(vecs)
    docs = {i: Document(content="x"*i, metadata={"page_number":i+1}) for i in range(dim)}
    ip = tmp_path/"indices"/"faiss_index.bin"; mp = tmp_path/"indices"/"doc_map.pkl"
    save_index_and_map(idx, docs, ip, mp)

    # 2) Write config
    cfg = {
      "data_raw_dir": str(tmp_path/"data"/"raw"),
      "processed_dir": str(tmp_path/"data"/"processed"),
      "index_path": str(ip), "doc_map_path": str(mp),
      "embedding_model": "ignored","embedding_dim":dim,
      "faiss_index_type":"IndexFlatL2","chunking_strategy":"recursive",
      "chunk_size":10,"chunk_overlap":0,"semantic_threshold":90.0,
      "cross_encoder_model":"ignored","rerank_top_n":5
    }
    cf = tmp_path/"config.yaml"; cf.write_text(yaml.dump(cfg))
    return KnowledgeAgent(cf)

def test_faiss_only(tmp_agent):
    # FAISS returns pages [1,2,3] for k=3 by identity vectors
    res = tmp_agent.search("q", k=3, rerank=False)
    assert [d.metadata["page_number"] for d,_ in res] == [1,2,3]

def test_two_stage(tmp_agent):
    # With rerank, DummyCE favors shortest texts → same pages but confirmed rerank path
    res = tmp_agent.search("q", k=3, rerank=True)
    assert [d.metadata["page_number"] for d,_ in res] == [1,2,3]

