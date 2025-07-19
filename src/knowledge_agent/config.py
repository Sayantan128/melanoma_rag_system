import yaml
from pathlib import Path

class Config:
    """
    Loads and validates parameters from a YAML config file.
    """
    def __init__(self, config_path: Path):
        self.config_path = Path(config_path)
        with open(self.config_path) as f:
            cfg = yaml.safe_load(f)

        # Required paths
        self.data_raw_dir = Path(cfg["data_raw_dir"])
        self.processed_dir = Path(cfg["processed_dir"])
        self.index_path = Path(cfg["index_path"])
        self.doc_map_path = Path(cfg["doc_map_path"])

        # Vectorization
        self.embedding_model = cfg["embedding_model"]
        self.embedding_dim = cfg["embedding_dim"]
        self.faiss_index_type= cfg["faiss_index_type"]

        # Chunking
        self.chunking_strategy = cfg.get("chunking_strategy", "semantic")
        self.chunk_size = cfg.get("chunk_size", 512)
        self.chunk_overlap = cfg.get("chunk_overlap", 50)
        self.semantic_threshold= cfg.get("semantic_threshold", 95.0)
        self.cross_encoder_model = cfg.get("cross_encoder_model")
        self.rerank_top_n = cfg.get("rerank_top_n", 50)

        # Basic validation
        if not self.data_raw_dir.exists():
            raise ValueError(f"data_raw_dir does not exist: {self.data_raw_dir}")

