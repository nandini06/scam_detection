from pathlib import Path
import pickle

import faiss
import numpy as np
import pandas as pd


def embeddings_to_matrix(df: pd.DataFrame, embedding_col: str = "embedding") -> np.ndarray:
    if embedding_col not in df.columns:
        raise ValueError(f"Missing embedding column: {embedding_col}")

    embeddings = np.array(df[embedding_col].tolist()).astype("float32")

    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embedding matrix, got shape {embeddings.shape}")

    return embeddings


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build FAISS index using inner product similarity.

    This assumes embeddings are already normalized.
    For normalized embeddings, inner product is equivalent to cosine similarity.
    """
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return index


def save_faiss_index(index: faiss.Index, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(output_path))

    print(f"\nSaved FAISS index to:")
    print(f" - {output_path}")


def load_faiss_index(index_path: Path) -> faiss.Index:
    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")

    return faiss.read_index(str(index_path))


def save_index_metadata(df: pd.DataFrame, output_path: Path) -> None:
    """
    Save chunk metadata in the same row order as FAISS index vectors.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata_cols = [
        col for col in [
            "chunk_id",
            "conversation_id",
            "start_step",
            "end_step",
            "chunk_text",
            "chunk_final_label",
            "chunk_features_str",
            "chunk_contexts_str",
            "all_labels_str",
        ]
        if col in df.columns
    ]

    metadata_df = df[metadata_cols].copy()
    metadata_df.to_parquet(output_path, index=False)

    print(f"\nSaved FAISS metadata to:")
    print(f" - {output_path}")


def load_index_metadata(metadata_path: Path) -> pd.DataFrame:
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    return pd.read_parquet(metadata_path)


def search_index(
    query_embedding: np.ndarray,
    index: faiss.Index,
    metadata_df: pd.DataFrame,
    top_k: int = 5,
) -> pd.DataFrame:
    """
    Search FAISS index and return top-k metadata rows with similarity scores.
    """
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)

    query_embedding = query_embedding.astype("float32")

    scores, indices = index.search(query_embedding, top_k)

    results = []

    for rank, idx in enumerate(indices[0]):
        if idx == -1:
            continue

        row = metadata_df.iloc[idx].to_dict()
        row["retrieval_rank"] = rank + 1
        row["similarity_score"] = float(scores[0][rank])
        results.append(row)

    return pd.DataFrame(results)