from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Data
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

# Artifacts
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
SPLITS_DIR = ARTIFACTS_DIR / "splits"
MODELS_ARTIFACTS_DIR = ARTIFACTS_DIR / "models"
METRIC_ARTIFACTS_DIR = ARTIFACTS_DIR / "metric"