import os
import warnings

warnings.filterwarnings("ignore", message="Tried to instantiate class")

os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"
os.environ["CHROMA_DO_NOT_TRACK"] = "true"
os.environ["POSTHOG_DISABLED"] = "true"

import logging
logging.getLogger("chromadb").setLevel(logging.ERROR)

try:
    import posthog
    posthog.capture = lambda *args, **kwargs: None
except ImportError:
    pass

from dotenv import load_dotenv
load_dotenv()

CHUNK_SIZE = 500
CHUNK_OVERLAP = 0
TOP_K_CHUNKS = 3
SIMILARITY_THRESHOLD = 0.3
PROMPT_VERSION = "langchain_default_v1"

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.1-8b-instant"
JUDGE_MODEL = "llama-3.3-70b-versatile"

CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OBSERVABILITY_DB = os.path.join(os.path.dirname(__file__), "observability.db")
