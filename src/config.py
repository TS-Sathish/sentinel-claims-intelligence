from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()
ROOT = Path(__file__).resolve().parents[1]
def configured_path(key, default):
    path = Path(os.getenv(key, default))
    return path if path.is_absolute() else ROOT / path

APP_NAME = os.getenv("APP_NAME", "Sentinel Claims Intelligence")
DATASET_PATH = configured_path("DATASET_PATH", "data/raw/fraud_oracle.csv")
MODEL_DIR = configured_path("MODEL_DIR", "models")
TARGET_COLUMN = os.getenv("TARGET_COLUMN", "").strip() or None
RISK_REVIEW_THRESHOLD = float(os.getenv("RISK_REVIEW_THRESHOLD", "0.65"))
if not 0 < RISK_REVIEW_THRESHOLD <= 1:
    raise ValueError("RISK_REVIEW_THRESHOLD must be greater than 0 and at most 1.")
