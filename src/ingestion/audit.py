import pandas as pd

def audit_dataframe(df: pd.DataFrame) -> None:
    print("\n===== DATAFRAME AUDIT =====")
    print(f"Shape: {df.shape}")

    print("\nColumns:")
    for col in df.columns:
        print(f" - {col}")

    print("\nDtypes:")
    print(df.dtypes)

    print("\nMissing values per column:")
    print(df.isna().sum())

    print("\nSample rows:")
    print(df.head(10))

    print("\nBasic text preview from first few rows:")
    for i, row in df.head(5).iterrows():
        print(f"\nRow index: {i}")
        for col in df.columns:
            value = row[col]
            value_str = str(value)
            if len(value_str) > 120:
                value_str = value_str[:120] + "..."
            print(f"{col}: {value_str}")


def audit_conversations(df: pd.DataFrame) -> None:
    print("\n===== CONVERSATION AUDIT =====")

    conv_counts = df.groupby("conversation_id").size().sort_values()
    print("\nTurns per conversation:")
    print(conv_counts.describe())

    print("\nSample conversation lengths:")
    print(conv_counts.head(10))

    if "label" in df.columns:
        print("\nRow-level label distribution:")
        print(df["label"].value_counts(dropna=False))

    if "conversation_id" in df.columns and "label" in df.columns:
        conv_label_summary = (
            df.groupby("conversation_id")["label"]
            .agg(lambda x: list(x))
            .head(5)
        )
        print("\nSample labels per conversation:")
        for conv_id, labels in conv_label_summary.items():
            print(f"Conversation {conv_id}: {labels}")


def audit_text_structure(df: pd.DataFrame) -> None:
    print("\n===== TEXT STRUCTURE AUDIT =====")

    if "has_step_marker" in df.columns:
        print("\nRows with [Step:n] marker:")
        print(df["has_step_marker"].value_counts(dropna=False))

    if "current_text" in df.columns:
        current_lengths = df["current_text"].apply(lambda x: len(str(x).split()))
        print("\nCurrent text word length stats:")
        print(current_lengths.describe())

    if "paired_response_text" in df.columns:
        response_lengths = df["paired_response_text"].apply(lambda x: len(str(x).split()))
        print("\nPaired response text word length stats:")
        print(response_lengths.describe())


def audit_label_quality(df: pd.DataFrame) -> None:
    if "label" not in df.columns:
        return

    print("\n===== LABEL QUALITY AUDIT =====")
    total = len(df)
    valid = int((df["label"] != "").sum())
    invalid = int((df["label"] == "").sum())

    print(f"Total rows: {total}")
    print(f"Valid normalized labels: {valid}")
    print(f"Invalid/unmapped labels: {invalid}")

    print("\nNormalized label distribution:")
    print(df["label"].replace("", "<INVALID>").value_counts(dropna=False))

    if "label_raw" in df.columns:
        invalid_rows = df[df["label"] == ""]
        if len(invalid_rows) > 0:
            print("\nTop invalid raw labels:")
            print(invalid_rows["label_raw"].value_counts().head(20))


def inspect_sample_conversation(df: pd.DataFrame, conversation_id: int) -> None:
    print(f"\n===== SAMPLE CONVERSATION: {conversation_id} =====")

    sample = (
        df[df["conversation_id"] == conversation_id]
        .sort_values("conversation_step")
        .copy()
    )

    for _, row in sample.iterrows():
        print(f"\nStep {row['conversation_step']}")
        print(f"Label raw: {row.get('label_raw', '')}")
        print(f"Label normalized: {row.get('label', '')}")
        print(f"Has [Step:n] marker: {row.get('has_step_marker', False)}")
        print(f"Current text: {row.get('current_text', '')}")
        print(f"Paired response text: {row.get('paired_response_text', '')}")
