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

BASE_MODEL = os.getenv("BASE_MODEL", "HuggingFaceTB/SmolLM2-135M-Instruct")
# Pinned revision for the default model (set BASE_MODEL_REVISION if you change models)
BASE_MODEL_REVISION = os.getenv("BASE_MODEL_REVISION") or "12fd25f77366fa6b3b4b768ec3050bf629380bac"
ADAPTER_PATH = ROOT / os.getenv("ADAPTER_PATH", "artifacts/support-lora-adapter")

MAX_SEQUENCE_LENGTH = int(os.getenv("MAX_SEQUENCE_LENGTH", "256"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "80"))
TRAIN_EPOCHS = int(os.getenv("TRAIN_EPOCHS", "8"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "0.0002"))
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))