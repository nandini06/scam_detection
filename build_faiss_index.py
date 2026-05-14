from pathlib import Path
import pandas as pd

from src.rag.vectordb import (
    embeddings_to_matrix,
    build_faiss_index,
    save_faiss_index,
    save_index_metadata,
)


INPUT_PATH = Path("outputs/embedded_train_chunks.parquet")

INDEX_OUTPUT_PATH = Path("outputs/faiss/train_chunks.index")
METADATA_OUTPUT_PATH = Path("outputs/faiss/train_chunks_metadata.parquet")


def main() -> None:
    print(f"\nReading embedded chunks from:")
    print(f" - {INPUT_PATH.resolve()}")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {INPUT_PATH}. Run embed_chunks.py first."
        )

    df = pd.read_parquet(INPUT_PATH)
    print(f"\nLoaded embedded dataframe shape: {df.shape}")

    embeddings = embeddings_to_matrix(df, embedding_col="embedding")
    print(f"Embedding matrix shape: {embeddings.shape}")

    index = build_faiss_index(embeddings)
    print(f"FAISS index built successfully.")
    print(f"Total vectors in index: {index.ntotal}")

    save_faiss_index(index, INDEX_OUTPUT_PATH)
    save_index_metadata(df, METADATA_OUTPUT_PATH)


if __name__ == "__main__":
    main()