import yaml
from pydantic import BaseModel


class Config(BaseModel):
    data_raw_dir: str
    processed_dir: str
    index_path: str
    doc_map_path: str
    embedding_model: str
    embedding_dim: int
    faiss_index_type: str
    chunking_strategy: str
    chunk_size: int
    chunk_overlap: int
    semantic_threshold: float
    cross_encoder_model: str
    rerank_top_n: int

    class Config:
        extra = "ignore"


def load_config(path: str) -> Config:
    with open(path, "r") as f:
        obj = yaml.safe_load(f)
    return Config(**obj)
