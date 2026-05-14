from pathlib import Path
import pandas as pd

from .config import OUTPUT_DIR


def load_raw_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found at: {path.resolve()}")

    print(f"Loading data from: {path.resolve()}")
    df = pd.read_csv(path)

    print(f"Loaded dataframe shape: {df.shape}")
    return df


def save_cleaned_rows(df: pd.DataFrame) -> None:
    #debug_csv = OUTPUT_DIR / "cleaned_rows_debug.csv"
    #debug_parquet = OUTPUT_DIR / "cleaned_rows_debug.parquet"

    model_csv = OUTPUT_DIR / "cleaned_rows_model.csv"
    model_parquet = OUTPUT_DIR / "cleaned_rows_model.parquet"

    # Full debug dump
    #df.to_csv(debug_csv, index=False)
    #df.to_parquet(debug_parquet, index=False)

    # Model-friendly subset
    model_columns = [
        col for col in [
            "conversation_id",
            "conversation_step",
            "current_text",
            "paired_response_text",
            "text_clean",
            "context",
            "label",
            "label_is_valid",
            "features_clean_str",
            "annotations",
        ]
        if col in df.columns
    ]

    model_df = df[model_columns].copy()

    model_df.to_csv(model_csv, index=False)
    model_df.to_parquet(model_parquet, index=False)

    print("\nSaved cleaned row datasets to:")
    #print(f" - {debug_csv}")
    #print(f" - {debug_parquet}")
    print(f" - {model_csv}")
    print(f" - {model_parquet}")


def save_conversations(conversations_df: pd.DataFrame) -> None:
    csv_path = OUTPUT_DIR / "conversations.csv"
    parquet_path = OUTPUT_DIR / "conversations.parquet"
    jsonl_path = OUTPUT_DIR / "conversations.jsonl"

    conversations_df.to_csv(csv_path, index=False)
    conversations_df.to_parquet(parquet_path, index=False)
    conversations_df.to_json(jsonl_path, orient="records", lines=True)

    print("\nSaved reconstructed conversations to:")
    print(f" - {csv_path}")
    print(f" - {parquet_path}")
    print(f" - {jsonl_path}")