from pathlib import Path
import pandas as pd

from src.rag.embedding import load_embedding_model, generate_embeddings
from src.rag.vectordb import load_faiss_index, load_index_metadata, search_index


INDEX_PATH = Path("outputs/faiss/train_chunks.index")
METADATA_PATH = Path("outputs/faiss/train_chunks_metadata.parquet")

MODEL_NAME = "all-MiniLM-L6-v2"


def main() -> None:
    model = load_embedding_model(MODEL_NAME)

    index = load_faiss_index(INDEX_PATH)
    metadata_df = load_index_metadata(METADATA_PATH)

    query_text = """
    I want to volunteer for a charity event and need registration details.
    """

    query_embedding = generate_embeddings(
        texts=[query_text],
        model=model,
        batch_size=1,
        normalize_embeddings=True,
    )

    results_df = search_index(
        query_embedding=query_embedding[0],
        index=index,
        metadata_df=metadata_df,
        top_k=5,
    )

    print("\n===== RETRIEVAL RESULTS =====")
    print(results_df[[
        "retrieval_rank",
        "similarity_score",
        "chunk_id",
        "chunk_final_label",
        "chunk_text",
    ]])


if __name__ == "__main__":
    main()