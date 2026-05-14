from pathlib import Path
import pandas as pd

from src.rag.embedding import load_embedding_model, generate_embeddings
from src.rag.vectordb import (
    load_faiss_index,
    load_index_metadata,
    search_index,
)


DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


def retrieve_similar_chunks(
    query_text: str,
    index_path: Path,
    metadata_path: Path,
    top_k: int = 5,
    model_name: str = DEFAULT_MODEL_NAME,
) -> pd.DataFrame:
    """
    Embed query text and retrieve top-k similar train chunks from FAISS.
    """

    model = load_embedding_model(model_name)

    index = load_faiss_index(index_path)
    metadata_df = load_index_metadata(metadata_path)

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
        top_k=top_k,
    )

    return results_df


def summarize_retrieved_labels(results_df: pd.DataFrame) -> dict:
    """
    Return retrieved label counts and proportions.
    """

    if results_df.empty or "chunk_final_label" not in results_df.columns:
        return {}

    counts = results_df["chunk_final_label"].value_counts().to_dict()
    total = len(results_df)

    proportions = {
        label: round(count / total, 4)
        for label, count in counts.items()
    }

    return {
        "counts": counts,
        "proportions": proportions,
    }


def format_retrieved_context(results_df: pd.DataFrame) -> str:
    """
    Format retrieved chunks for RAG prompt.
    """

    if results_df.empty:
        return "No retrieved examples available."

    context_blocks = []

    for _, row in results_df.iterrows():
        rank = row.get("retrieval_rank", "")
        score = row.get("similarity_score", "")
        label = row.get("chunk_final_label", "")
        chunk_id = row.get("chunk_id", "")
        text = row.get("chunk_text", "")

        block = f"""
Retrieved Example {rank}
Chunk ID: {chunk_id}
Label: {label}
Similarity Score: {score:.4f}

{text}
""".strip()

        context_blocks.append(block)

    return "\n\n---\n\n".join(context_blocks)