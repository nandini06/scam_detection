from pathlib import Path
import pandas as pd

from src.ingestion.config import OUTPUT_DIR
from src.chunking.chunking import create_chunks, audit_chunks, save_chunks


INPUT_PATH = OUTPUT_DIR / "cleaned_rows_model.parquet"
WINDOW_SIZE = 3
STRIDE = 1


def main() -> None:
    print(f"Reading cleaned rows from: {INPUT_PATH.resolve()}")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {INPUT_PATH}. Run ingest.py first."
        )

    df = pd.read_parquet(INPUT_PATH)
    print(f"Loaded cleaned rows: {df.shape}")

    chunks_df = create_chunks(
        df=df,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
    )

    audit_chunks(chunks_df)
    save_chunks(chunks_df, OUTPUT_DIR)


if __name__ == "__main__":
    main()