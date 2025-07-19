from pathlib import Path
import faiss
from src.knowledge_agent.config import Config
from src.knowledge_agent.ingestion import PDFExtractor, Document
from src.knowledge_agent.chunking import chunk_recursively, chunk_semantically
from src.knowledge_agent.vectorization import EmbeddingModel, save_index_and_map, load_index_and_map

class KnowledgeAgent:
    """
    Unified API for ingestion → chunking → vectorization → FAISS indexing,
    with a search(query, k) method for retrieval.
    """
    def __init__(self, config_path: str | Path):
        cfg = Config(Path(config_path))
        self.cfg = cfg

        # If index+map already exist, load them
        if cfg.index_path.exists() and cfg.doc_map_path.exists():
            self.index, self.doc_map = load_index_and_map(cfg.index_path, cfg.doc_map_path)
            return

        # Otherwise run the full build pipeline
        # 1) Ingest all PDFs
        docs = []
        for pdf_file in cfg.data_raw_dir.glob("*.pdf"):
            docs += PDFExtractor(pdf_file).extract_text_with_metadata()

        # 2) Chunk
        if cfg.chunking_strategy == "recursive":
            chunks = chunk_recursively(docs, cfg.chunk_size, cfg.chunk_overlap)
        else:
            chunks = chunk_semantically(docs,cfg.embedding_model,cfg.semantic_threshold)

        # 3) Build doc_map
        self.doc_map = {i: chunk for i, chunk in enumerate(chunks)}

        # 4) Embed
        emb_model = EmbeddingModel(cfg.embedding_model)
        texts = [chunk.content for chunk in chunks]
        embeddings = emb_model.encode(texts)

        # 5) FAISS index
        dim = embeddings.shape[1]
        if cfg.faiss_index_type == "IndexFlatL2":
            index = faiss.IndexFlatL2(dim)
        else:
            # fallback to flat
            index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        self.index = index

        # 6) Persist
        save_index_and_map(self.index, self.doc_map, cfg.index_path, cfg.doc_map_path)

    def search(self, query: str, k: int = 5) -> list[tuple[Document, float]]:
        """
        Encode `query`, run FAISS search, and return list of (Document, distance).
        """
        emb_model = EmbeddingModel(self.cfg.embedding_model)
        q_vec = emb_model.encode([query])
        distances, indices = self.index.search(q_vec, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            doc = self.doc_map[int(idx)]
            results.append((doc, float(dist)))
        return results

