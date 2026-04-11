import pandas as pd

from src.ingestion.config import RAW_DATA_PATH
from src.ingestion.audit import (
    audit_dataframe,
    audit_conversations,
    audit_text_structure,
    audit_label_quality,
    inspect_sample_conversation,
)
from src.ingestion.pipeline import (
    standardize_column_names,
    check_expected_columns,
    clean_rows,
)
from src.ingestion.io_utils import save_cleaned_rows, save_conversations
from src.ingestion.reconstruct import reconstruct_conversations


DEBUG = False


def main() -> None:
    print(f"Looking for file at: {RAW_DATA_PATH.resolve()}")

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"CSV file not found at {RAW_DATA_PATH}. "
            "Please download the Kaggle CSV and place it inside data/raw/."
        )

    df = pd.read_csv(RAW_DATA_PATH)
    df = standardize_column_names(df)
    check_expected_columns(df)

    print("\nBefore cleaning:")
    print(df.shape)

    df = clean_rows(df)

    print("\nAfter cleaning:")
    print(df.shape)

    save_cleaned_rows(df)

    if DEBUG:
        audit_dataframe(df)
        audit_conversations(df)
        audit_text_structure(df)
        audit_label_quality(df)
        inspect_sample_conversation(df, conversation_id=6)
        inspect_sample_conversation(df, conversation_id=2)
        inspect_sample_conversation(df, conversation_id=4)

    conversations_df = reconstruct_conversations(df)

    print("\n===== CONVERSATION-LEVEL DATA =====")
    print(conversations_df.head(5))
    print(f"Conversation dataset shape: {conversations_df.shape}")

    save_conversations(conversations_df)


if __name__ == "__main__":
    main()