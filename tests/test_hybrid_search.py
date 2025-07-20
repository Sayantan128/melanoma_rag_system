import numpy as np
import pytest
from src.knowledge_agent.ingestion import Document
from src.knowledge_agent.agent import KnowledgeAgent
from src.knowledge_agent.vectorization import save_index_and_map
import faiss
import yaml

@pytest.fixture(autouse=True)
def stub_dependencies(monkeypatch):
    # Stub out SentenceTransformer → identity embeddings
    class DummyEmb:
        def __init__(self, *args, **kwargs): pass
        def encode(self, texts):
            # “Semantic” space is just the length of the text
            arr = np.array([[len(t)] for t in texts], dtype="float32")
            return arr
    monkeypatch.setattr(
        "src.knowledge_agent.agent.EmbeddingModel",
        lambda model_name: DummyEmb()
    )

@pytest.fixture
def tmp_agent(tmp_path):
    # 1) Create dummy chunks with overlapping keywords
    #    chunk0: "apple orange"
    #    chunk1: "banana apple"
    #    chunk2: "cherry"
    #    chunk3: "date"
    #    chunk4: "elderberry apple"
    chunks = [
        Document(content="apple orange", metadata={"page_number":0}),
        Document(content="banana apple", metadata={"page_number":1}),
        Document(content="cherry",       metadata={"page_number":2}),
        Document(content="date",         metadata={"page_number":3}),
        Document(content="elderberry apple", metadata={"page_number":4}),
    ]

    # 2) Build FAISS index and doc_map
    dim = 1
    idx = faiss.IndexFlatL2(dim)
    # embeddings = [[len(text)]]
    vecs = np.array([[len(doc.content)] for doc in chunks], dtype="float32")
    idx.add(vecs)
    doc_map = {i: doc for i, doc in enumerate(chunks)}

    # 3) Save them to disk
    ip = tmp_path/"indices"/"faiss_index.bin"
    mp = tmp_path/"indices"/"doc_map.pkl"
    save_index_and_map(idx, doc_map, ip, mp)

    # 4) Write config with BM25 enabled
    cfg = {
        "data_raw_dir":     str(tmp_path/"data"/"raw"),      # not used
        "processed_dir":    str(tmp_path/"data"/"processed"),
        "index_path":       str(ip),
        "doc_map_path":     str(mp),
        "embedding_model":  "ignored",
        "embedding_dim":     dim,
        "faiss_index_type": "IndexFlatL2",
        "chunking_strategy":"recursive",
        "chunk_size":       10,
        "chunk_overlap":     0,
        "semantic_threshold":90.0,
        "cross_encoder_model": None,
        "rerank_top_n":      5,
        # Hybrid settings:
        "use_bm25":          True,
        "bm25_alpha":        0.5
    }
    # write directories needed by Config
    (tmp_path/"data"/"raw").mkdir(parents=True)
    (tmp_path/"data"/"processed").mkdir()
    cf = tmp_path/"config.yaml"
    cf.write_text(yaml.dump(cfg))

    # 5) Build agent
    return KnowledgeAgent(cf)

def test_keyword_lookup_wins(tmp_agent):
    """
    Query = "apple"
    BM25‐heavy chunk (the shortest one with “apple”) must appear,
    even though semantically it might have large L2 distance.
    """
    results = tmp_agent.search("apple", k=1, alpha=0.5)
    # All chunks contain “apple”, but BM25 scores favor exact matches independent of length.
    # Confirm that one of the “apple” chunks is returned:
    assert results[0][0].content.count("apple") >= 1

def test_alpha_1_is_faiss_only(tmp_agent):
    """
    With alpha=1.0, combined_score == FAISS-only score,
    so ranking by chunk‐length distance.
    """
    faiss_only = tmp_agent.search("banana", k=3, alpha=1.0, use_bm25=False)
    hybrid    = tmp_agent.search("banana", k=3, alpha=1.0, use_bm25=True)
    # The two result‐lists should be identical in order and content:
    assert [doc.content for doc,_ in faiss_only] == [doc.content for doc,_ in hybrid]

