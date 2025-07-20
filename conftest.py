import sys, os

# make “import src.knowledge_agent…” resolve to ./src/knowledge_agent
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))
