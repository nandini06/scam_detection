from pathlib import Path
import json
import pandas as pd

from sklearn.model_selection import train_test_split
from src.ingestion.config import OUTPUT_DIR


INPUT_PATH = OUTPUT_DIR / "chunks.parquet"
RANDOM_SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def validate_split_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-9:
        raise ValueError(
            f"Split ratios must sum to 1.0, but got {total:.6f}"
        )


def stratified_split_conversation_ids(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    random_seed: int,
):
    validate_split_ratios(train_ratio, val_ratio, test_ratio)

    if "conversation_id" not in df.columns:
        raise ValueError("Expected 'conversation_id' column.")

    if "chunk_final_label" not in df.columns:
        raise ValueError("Expected 'chunk_final_label' column.")

    # -------------------------------------------------
    # Build one label per conversation
    # -------------------------------------------------
    conversation_labels_df = (
        df.groupby("conversation_id")["chunk_final_label"]
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "uncertain")
        .reset_index()
    )

    conversation_ids = conversation_labels_df["conversation_id"]
    labels = conversation_labels_df["chunk_final_label"]

    # -------------------------------------------------
    # Train split
    # -------------------------------------------------
    train_ids, temp_ids, train_labels, temp_labels = train_test_split(
        conversation_ids,
        labels,
        test_size=(1.0 - train_ratio),
        stratify=labels,
        random_state=random_seed,
    )

    # -------------------------------------------------
    # Validation/Test split
    # -------------------------------------------------
    val_relative_ratio = val_ratio / (val_ratio + test_ratio)

    val_ids, test_ids = train_test_split(
        temp_ids,
        test_size=(1.0 - val_relative_ratio),
        random_state=random_seed,
    )

    return (
        sorted(train_ids.tolist()),
        sorted(val_ids.tolist()),
        sorted(test_ids.tolist()),
    )


def filter_by_conversation_ids(df: pd.DataFrame, conversation_ids: list) -> pd.DataFrame:
    return df[df["conversation_id"].isin(conversation_ids)].copy()


def audit_split(name: str, df: pd.DataFrame) -> None:
    print(f"\n===== {name.upper()} SPLIT AUDIT =====")
    print(f"Rows: {len(df)}")
    print(f"Unique conversations: {df['conversation_id'].nunique()}")

    if "chunk_final_label" in df.columns:
        print("\nLabel distribution:")
        label_counts = df["chunk_final_label"].value_counts(dropna=False)

        print(label_counts)

        print("\nNormalized distribution:")
        print(
            (
                label_counts / label_counts.sum()
            ).round(4)
        )

    if "chunk_word_length" in df.columns:
        print("\nChunk word length stats:")
        print(df["chunk_word_length"].describe())


def save_split(df: pd.DataFrame, split_name: str) -> None:
    csv_path = OUTPUT_DIR / f"{split_name}_chunks.csv"
    parquet_path = OUTPUT_DIR / f"{split_name}_chunks.parquet"

    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)

    print(f"\nSaved {split_name} split to:")
    print(f" - {csv_path}")
    print(f" - {parquet_path}")


def save_split_metadata(train_ids: list, val_ids: list, test_ids: list) -> None:
    metadata = {
        "random_seed": RANDOM_SEED,
        "train_ratio": TRAIN_RATIO,
        "val_ratio": VAL_RATIO,
        "test_ratio": TEST_RATIO,
        "train_conversation_ids": train_ids,
        "val_conversation_ids": val_ids,
        "test_conversation_ids": test_ids,
    }

    output_path = OUTPUT_DIR / "split_conversation_ids.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved split metadata to:")
    print(f" - {output_path}")


def main() -> None:
    print(f"Reading chunks from: {INPUT_PATH.resolve()}")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {INPUT_PATH}. Run chunk_data.py first."
        )

    df = pd.read_parquet(INPUT_PATH)
    print(f"Loaded chunks dataframe shape: {df.shape}")

    if "conversation_id" not in df.columns:
        raise ValueError("Expected 'conversation_id' column in chunks dataframe.")

    unique_conversation_ids = sorted(df["conversation_id"].dropna().unique().tolist())
    print(f"Total unique conversations: {len(unique_conversation_ids)}")

    train_ids, val_ids, test_ids = stratified_split_conversation_ids(
        df=df,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        random_seed=RANDOM_SEED,
    )

    print("\nConversation counts by split:")
    print(f"Train: {len(train_ids)}")
    print(f"Val:   {len(val_ids)}")
    print(f"Test:  {len(test_ids)}")

    train_df = filter_by_conversation_ids(df, train_ids)
    val_df = filter_by_conversation_ids(df, val_ids)
    test_df = filter_by_conversation_ids(df, test_ids)

    audit_split("train", train_df)
    audit_split("val", val_df)
    audit_split("test", test_df)

    save_split(train_df, "train")
    save_split(val_df, "val")
    save_split(test_df, "test")

    save_split_metadata(train_ids, val_ids, test_ids)


if __name__ == "__main__":
    main()