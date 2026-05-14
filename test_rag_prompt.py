from pathlib import Path

from src.rag.retrieval import retrieve_similar_chunks
from src.rag.prompts import build_rag_prompt


INDEX_PATH = Path("outputs/faiss/train_chunks.index")
METADATA_PATH = Path("outputs/faiss/train_chunks_metadata.parquet")


def main() -> None:
    query_text = """
    The caller says they are from the tax department and that immediate payment is required
    to avoid legal action. They refuse to provide official written documentation.
    """

    retrieved_df = retrieve_similar_chunks(
        query_text=query_text,
        index_path=INDEX_PATH,
        metadata_path=METADATA_PATH,
        top_k=5,
    )

    print("\n===== RETRIEVED CHUNKS =====")
    print(
        retrieved_df[
            [
                "retrieval_rank",
                "similarity_score",
                "chunk_id",
                "chunk_final_label",
            ]
        ]
    )

    prompt = build_rag_prompt(
        query_text=query_text,
        retrieved_df=retrieved_df,
    )

    print("\n===== RAG PROMPT =====")
    print(prompt)


if __name__ == "__main__":
    main()