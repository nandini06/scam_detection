from pathlib import Path
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# ---------------------------------------------------
# DEFAULT EMBEDDING MODEL
# ---------------------------------------------------

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------

def load_embedding_model(
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> SentenceTransformer:
    """
    Load sentence-transformer embedding model.
    """

    print(f"\nLoading embedding model: {model_name}")

    model = SentenceTransformer(model_name)

    print("Embedding model loaded successfully.")

    return model


# ---------------------------------------------------
# GENERATE EMBEDDINGS
# ---------------------------------------------------

def generate_embeddings(
    texts: list[str],
    model: SentenceTransformer,
    batch_size: int = 32,
    normalize_embeddings: bool = True,
) -> np.ndarray:
    """
    Generate embeddings for a list of texts.
    """

    print(f"\nGenerating embeddings for {len(texts)} texts...")

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize_embeddings,
    )

    print(f"Generated embeddings shape: {embeddings.shape}")

    return embeddings


# ---------------------------------------------------
# EMBED CHUNKS DATAFRAME
# ---------------------------------------------------

def embed_chunks_dataframe(
    df: pd.DataFrame,
    text_column: str,
    model: SentenceTransformer,
    batch_size: int = 32,
) -> pd.DataFrame:
    """
    Generate embeddings for dataframe text column.
    """

    if text_column not in df.columns:
        raise ValueError(
            f"Column '{text_column}' not found in dataframe."
        )

    work_df = df.copy()

    texts = (
        work_df[text_column]
        .fillna("")
        .astype(str)
        .tolist()
    )

    embeddings = generate_embeddings(
        texts=texts,
        model=model,
        batch_size=batch_size,
    )

    # store as python lists for parquet/json compatibility
    work_df["embedding"] = embeddings.tolist()

    work_df["embedding_dimension"] = embeddings.shape[1]

    return work_df


# ---------------------------------------------------
# SAVE EMBEDDED DATAFRAME
# ---------------------------------------------------

def save_embedded_dataframe(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Save embedded dataframe.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(output_path, index=False)

    print(f"\nSaved embedded dataframe to:")
    print(f" - {output_path}")


# ---------------------------------------------------
# AUDIT EMBEDDINGS
# ---------------------------------------------------

def audit_embeddings(df: pd.DataFrame) -> None:
    """
    Basic embedding audit.
    """

    print("\n===== EMBEDDING AUDIT =====")

    print(f"Shape: {df.shape}")

    if "embedding" not in df.columns:
        print("No embedding column found.")
        return

    print("\nSample embedding info:")

    sample_embedding = df["embedding"].iloc[0]

    print(f"Embedding type: {type(sample_embedding)}")
    print(f"Embedding dimension: {len(sample_embedding)}")

    if "chunk_text" in df.columns:
        print("\nSample chunk:")
        print(df["chunk_text"].iloc[0][:500])

    print("\nSample embedding values:")
    print(sample_embedding[:10])