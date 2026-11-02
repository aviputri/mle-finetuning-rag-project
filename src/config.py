import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

KB_DIR = ROOT / "data" / "knowledge_base"
TRAIN_FILE = ROOT / "data" / "training" / "instruction_examples.jsonl"
EVAL_FILE = ROOT / "data" / "evaluation" / "cases.json"

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
# Pinned revision for the default model (set EMBEDDING_REVISION if you change models)
EMBEDDING_REVISION = os.getenv("EMBEDDING_REVISION") or "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
TOP_K = int(os.getenv("TOP_K", "3"))
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.30"))