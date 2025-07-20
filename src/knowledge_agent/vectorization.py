from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import pickle
from pathlib import Path
from src.knowledge_agent.ingestion import Document

class EmbeddingModel:
    """Wraps a sentence-transformers model for FAISS-ready embeddings."""
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def get_embedding_dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str], show_progress_bar: bool = False) -> np.ndarray:
        embeddings = self.model.encode(texts, show_progress_bar=show_progress_bar)
        return embeddings.astype("float32")


def save_index_and_map(
    index: faiss.Index,
    doc_map: dict[int, Document],
    index_path: Path,
    map_path: Path
) -> None:
    """Serialize FAISS index and the doc_map to disk."""
    index_path.parent.mkdir(parents=True, exist_ok=True)
    map_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    with open(map_path, "wb") as f:
        pickle.dump(doc_map, f)


def load_index_and_map(
    index_path: Path,
    map_path: Path
) -> tuple[faiss.Index, dict[int, Document]]:
    """Load FAISS index and doc_map from disk."""
    if not index_path.exists() or not map_path.exists():
        raise FileNotFoundError("Index or doc_map file not found.")
    index = faiss.read_index(str(index_path))
    with open(map_path, "rb") as f:
        doc_map = pickle.load(f)
    return index, doc_map
