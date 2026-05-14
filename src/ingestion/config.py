from pathlib import Path

RAW_DATA_PATH = Path("data/raw/BETTER31.csv")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VALID_LABELS = {
    "safe",
    "suspicious",
    "scam",
    "uncertain",
}

STEP_MARKER_PATTERN = r"\[Step:\s*\d+\]"