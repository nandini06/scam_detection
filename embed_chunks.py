from pathlib import Path
import pandas as pd

from src.rag.embedding import (
    load_embedding_model,
    embed_chunks_dataframe,
    save_embedded_dataframe,
    audit_embeddings,
)


# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------

INPUT_PATH = Path("outputs/train_chunks.parquet")

OUTPUT_PATH = Path(
    "outputs/embedded_train_chunks.parquet"
)

TEXT_COLUMN = "chunk_text"

MODEL_NAME = "all-MiniLM-L6-v2"

BATCH_SIZE = 32


# ---------------------------------------------------
# MAIN
# ---------------------------------------------------

def main() -> None:

    print(f"\nReading chunks from:")
    print(f" - {INPUT_PATH.resolve()}")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {INPUT_PATH}"
        )

    df = pd.read_parquet(INPUT_PATH)

    print(f"\nLoaded dataframe shape: {df.shape}")

    model = load_embedding_model(MODEL_NAME)

    embedded_df = embed_chunks_dataframe(
        df=df,
        text_column=TEXT_COLUMN,
        model=model,
        batch_size=BATCH_SIZE,
    )

    audit_embeddings(embedded_df)

    save_embedded_dataframe(
        embedded_df,
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()