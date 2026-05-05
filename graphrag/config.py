import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

LM_STUDIO_URL = os.getenv("LM_STUDIO_URL", "http://127.0.0.1:1234")
LM_STUDIO_CHAT_MODEL = os.getenv("LM_STUDIO_CHAT_MODEL", "openai/gpt-oss-20b")
LM_STUDIO_EMBED_MODEL = os.getenv("LM_STUDIO_EMBED_MODEL", "text-embedding-nomic-embed-text-v1.5")

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

PDF_DIR = os.getenv("PDF_DIR", "data/pdfs")
CHUNKS_PATH = os.getenv("CHUNKS_PATH", "data/chunks.json")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))


def get_chat_client() -> tuple[OpenAI, str]:
    """Return (client, model_name). Uses OpenAI if key present, else LM Studio."""
    if OPENAI_API_KEY:
        return OpenAI(api_key=OPENAI_API_KEY), OPENAI_MODEL
    print("[config] No OPENAI_API_KEY found — using LM Studio fallback.")
    return OpenAI(base_url=f"{LM_STUDIO_URL}/v1", api_key="lm-studio"), LM_STUDIO_CHAT_MODEL


def get_embed_client() -> tuple[OpenAI, str]:
    """Return (client, model_name) for embeddings.
    Uses OpenAI text-embedding if API key present, otherwise LM Studio local model."""
    if OPENAI_API_KEY:
        return OpenAI(api_key=OPENAI_API_KEY), OPENAI_EMBED_MODEL
    return OpenAI(base_url=f"{LM_STUDIO_URL}/v1", api_key="lm-studio"), LM_STUDIO_EMBED_MODEL
