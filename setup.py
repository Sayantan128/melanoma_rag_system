# setup.py
from setuptools import setup, find_packages

setup(
    name="knowledge_agent",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        # your runtime deps here, e.g.
        "faiss-cpu>=1.8.0",
        "sentence-transformers>=2.7.0",
        "rank-bm25>=0.2.2",
        # …
    ],
)

