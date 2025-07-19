from pathlib import Path
import faiss
from src.knowledge_agent.config import Config
from src.knowledge_agent.ingestion import PDFExtractor, Document
from src.knowledge_agent.chunking import chunk_recursively, chunk_semantically
from src.knowledge_agent.vectorization import EmbeddingModel, save_index_and_map, load_index_and_map
from sentence_transformers import CrossEncoder

class KnowledgeAgent:
    """
    Unified API for ingestion → chunking → vectorization → FAISS indexing,
    with a search(query, k) method for retrieval.
    """
    def __init__(self, config_path: str | Path):
        cfg = Config(Path(config_path))
        self.cfg = cfg

        # 1) If index+map exist, load them
        if cfg.index_path.exists() and cfg.doc_map_path.exists():
            self.index, self.doc_map = load_index_and_map(cfg.index_path, cfg.doc_map_path)
            return

        # 2) Ingest
        docs = []
        for pdf_file in cfg.data_raw_dir.glob("*.pdf"):
            docs += PDFExtractor(pdf_file).extract_text_with_metadata()

        # 3) Chunk
        if cfg.chunking_strategy == "recursive":
            chunks = chunk_recursively(docs, cfg.chunk_size, cfg.chunk_overlap)
        else:
            chunks = chunk_semantically(docs, cfg.embedding_model, cfg.semantic_threshold)

        # 4) Build doc_map
        self.doc_map = {i: chunk for i, chunk in enumerate(chunks)}

        # 5) Embed
        emb_model = EmbeddingModel(cfg.embedding_model)
        texts = [chunk.content for chunk in chunks]
        embeddings = emb_model.encode(texts)

        # 6) Index
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim) if cfg.faiss_index_type == "IndexFlatL2" else faiss.IndexFlatL2(dim)
        index.add(embeddings)
        self.index = index

        # 7) Persist
        save_index_and_map(self.index, self.doc_map, cfg.index_path, cfg.doc_map_path)

    def search(
        self,
        query: str,
        k: int = 5,
        rerank: bool = True
    ) -> list[tuple[Document, float]]:
        """
        Stage 1: FAISS search for self.cfg.rerank_top_n candidates.
        Stage 2: Cross-encoder re-rank to pick top k if rerank=True.
        """
        # FAISS first pass
        emb_model = EmbeddingModel(self.cfg.embedding_model)
        q_vec = emb_model.encode([query])
        top_n = max(k, self.cfg.rerank_top_n)
        distances, indices = self.index.search(q_vec, top_n)

        candidates = [
            (self.doc_map[int(idx)], float(dist))
            for idx, dist in zip(indices[0], distances[0])
        ]

        # Skip reranking?
        if not rerank or not self.cfg.cross_encoder_model:
            return candidates[:k]

        # Cross-encoder second pass
        if not hasattr(self, "_cross_encoder"):
            self._cross_encoder = CrossEncoder(self.cfg.cross_encoder_model)

        pairs = [[query, doc.content] for doc, _ in candidates]
        scores = self._cross_encoder.predict(pairs)

        reranked = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True
        )
        return [(doc, float(score)) for ((doc, _), score) in reranked[:k]]
