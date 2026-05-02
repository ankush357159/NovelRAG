import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


# Find the first directory upwards that contains 'pyproject.toml'
def get_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return current.parent

BASE_DIR = get_project_root()
ASSETS_DIR = BASE_DIR / "assets"
PROCESSED_DIR = ASSETS_DIR / "processed"
CHUNK_DIR = PROCESSED_DIR / "chunk-document"

# Create the folders if they don't exist yet
# parents=True creates 'assets' AND 'processed' in one go
# exist_ok=True prevents an error if the folder is already there
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
CHUNK_DIR.mkdir(parents=True, exist_ok=True)

# --- Routing ---
# L2 relevance = 1 - distance / sqrt(2).  For normalized MiniLM vectors a
# cosine similarity of ~0.75 maps to an L2 relevance of ~0.50, so use 0.45
# as a conservative default.  Override via the ROUTING_THRESHOLD env var.
ROUTING_THRESHOLD: float = float(os.getenv("ROUTING_THRESHOLD", "0.45"))

# --- Validate required env vars at startup ---
_REQUIRED = ["OPENAI_API_KEY"]
_missing = [v for v in _REQUIRED if not os.getenv(v)]
if _missing:
    raise RuntimeError(f"Missing required environment variables: {', '.join(_missing)}")