import pickle
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
import yaml

from src.knowledge_agent.ingestion import Document, PDFExtractor
from src.knowledge_agent.vectorization import (
    EmbeddingModel,
    save_index_and_map,
    load_index_and_map,
)

# Expose CrossEncoder so tests can monkey-patch it
try:
    from sentence_transformers import CrossEncoder
except ImportError:
    class CrossEncoder:
        def __init__(self, *args, **kwargs): pass
        def predict(self, pairs):
            return np.zeros(len(pairs), dtype="float32")


class Config:
    """
    Simple config loader: turns a dict into attributes.
    """
    def __init__(self, **data):
        for k, v in data.items():
            setattr(self, k, v)


class KnowledgeAgent:
    def __init__(self, config_path: Path):
        # load config
        cfg_dict = yaml.safe_load(Path(config_path).read_text())
        self.cfg = Config(**cfg_dict)

        # 0) if a prebuilt index exists on disk, just load it
        idx_path = Path(self.cfg.index_path)
        map_path = Path(self.cfg.doc_map_path)
        if idx_path.exists() and map_path.exists():
            self.index, self.doc_map = load_index_and_map(idx_path, map_path)
        else:
            # 1) ingest / chunk
            extractor = PDFExtractor(Path(self.cfg.data_raw_dir))
            docs = extractor.extract_text_with_metadata()  # List[Document]
            from src.knowledge_agent.chunking import (
                chunk_recursively,
                chunk_semantically,
            )
            if self.cfg.chunking_strategy == "recursive":
                chunks = chunk_recursively(
                    docs,
                    chunk_size=self.cfg.chunk_size,
                    chunk_overlap=self.cfg.chunk_overlap,
                )
            else:
                chunks = chunk_semantically(
                    docs,
                    model_name=self.cfg.embedding_model,
                    threshold=self.cfg.semantic_threshold,
                )

            # 2) build FAISS
            texts = [c.content for c in chunks]
            em_model = EmbeddingModel(self.cfg.embedding_model)
            vectors = em_model.encode(texts)  # shape (N, dim)
            self.index = getattr(faiss, self.cfg.faiss_index_type)(
                em_model.get_embedding_dimension()
            )
            self.index.add(np.array(vectors, dtype="float32"))
            self.doc_map = {i: chunks[i] for i in range(len(chunks))}

            # persist for future runs
            save_index_and_map(self.index, self.doc_map, idx_path, map_path)

        # 3) build BM25
        tokenized = [doc.content.split() for doc in self.doc_map.values()]
        self._bm25 = BM25Okapi(tokenized)
        self._bm25_map = {
            i: doc for i, doc in enumerate(self.doc_map.values())
        }

    def search(
        self,
        query: str,
        k: int = 5,
        rerank: bool = True,
        alpha: float = 0.7,
        use_bm25: bool = None,
    ) -> List[Tuple[Document, float]]:
        # default use_bm25 from config
        if use_bm25 is None:
            use_bm25 = getattr(self.cfg, "use_bm25", False)

        # --- FAISS pass ---
        q_vec = EmbeddingModel(self.cfg.embedding_model).encode([query])
        top_n = max(k, getattr(self.cfg, "rerank_top_n", k))
        dists, ids = self.index.search(np.array(q_vec, dtype="float32"), top_n)
        faiss_hits = [
            (self.doc_map[i], float(d))
            for i, d in zip(ids[0], dists[0])
            if i >= 0
        ]

        # no rerank or BM25 disabled → return FAISS‐only
        if not rerank or not use_bm25:
            return faiss_hits[:k]

        # --- BM25 pass ---
        tokens = query.split()
        bm25_scores = self._bm25.get_scores(tokens)
        bm25_hits = sorted(
            ((self._bm25_map[i], bm25_scores[i]) for i in range(len(bm25_scores))),
            key=lambda x: x[1],
            reverse=True,
        )[: self.cfg.rerank_top_n]

        # --- normalize & combine ---
        candidates = {doc for doc, _ in faiss_hits} | {doc for doc, _ in bm25_hits}
        faiss_dict = {doc: dist for doc, dist in faiss_hits}
        bm25_dict  = {doc: score for doc, score in bm25_hits}

        fvals = np.array(list(faiss_dict.values()), dtype=np.float32) if faiss_dict else np.array([0], dtype=np.float32)
        bvals = np.array(list(bm25_dict.values()), dtype=np.float32) if bm25_dict else np.array([0], dtype=np.float32)

        fmin, fmax = fvals.min(), fvals.max()
        bmin, bmax = bvals.min(), bvals.max()

        out: List[Tuple[Document, float]] = []
        for doc in candidates:
            fd = faiss_dict.get(doc, fmax)
            bs = bm25_dict.get(doc, bmin)
            fn = 1 - (fd - fmin) / (fmax - fmin + 1e-12)
            bn = (bs - bmin) / (bmax - bmin + 1e-12)
            score = alpha * fn + (1 - alpha) * bn
            out.append((doc, score))

        out.sort(key=lambda x: x[1], reverse=True)
        return out[:k]
