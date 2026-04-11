import pandas as pd

from .cleaners import (
    clean_text_basic,
    normalize_feature_name,
    normalize_label,
    parse_feature_list
)

from .parsers import (
    has_step_marker,
    split_exchange_text
)


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )
    return df


def check_expected_columns(df: pd.DataFrame) -> None:
    expected_candidates = [
        "conversation_id",
        "conversation_step",
        "text",
        "context",
        "label",
        "features",
        "annotations",
    ]

    print("\n===== EXPECTED COLUMN CHECK =====")
    for col in expected_candidates:
        status = "FOUND" if col in df.columns else "NOT FOUND"
        print(f"{col}: {status}")


def clean_rows(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Light clean all object columns
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].apply(clean_text_basic)

    # Drop fully empty rows
    df = df.dropna(how="all")

    # Parse transcript structure
    if "text" in df.columns:
        df["text_raw"] = df["text"]
        df["has_step_marker"] = df["text"].apply(has_step_marker)

        parsed = df["text"].apply(split_exchange_text)
        df["current_text"] = parsed.apply(lambda x: x[0])
        df["paired_response_text"] = parsed.apply(lambda x: x[1])

        # keep a non-destructive cleaned version of original row
        df["text_normalized"] = df["text"].apply(clean_text_basic)

        before = len(df)
        df = df[
            (df["current_text"].str.len() > 0) | (df["paired_response_text"].str.len() > 0)
        ].copy()
        after = len(df)

        print(f"\nDropped {before - after} rows with empty parsed text.")

    # Numeric step
    if "conversation_step" in df.columns:
        df["conversation_step"] = pd.to_numeric(df["conversation_step"], errors="coerce")

    # Normalize labels
    if "label" in df.columns:
        df["label_raw"] = df["label"]
        df["label"] = df["label"].apply(normalize_label)
        df["label_is_valid"] = df["label"].apply(lambda x: bool(x))

    # Normalize features
    if "features" in df.columns:
        df["features_list"] = df["features"].apply(parse_feature_list)
        df["features_clean_str"] = df["features_list"].apply(lambda xs: "|".join(xs))

    return df
