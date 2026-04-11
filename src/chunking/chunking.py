from pathlib import Path
import pandas as pd


def clean_list_values(values: list[object]) -> list[str]:
    cleaned = []
    for x in values:
        if pd.isna(x):
            continue
        x_str = str(x).strip()
        if not x_str or x_str.lower() == "nan":
            continue
        cleaned.append(x_str)
    return cleaned


def parse_pipe_separated_features(value: object) -> list[str]:
    if pd.isna(value):
        return []

    value = str(value).strip()
    if not value or value.lower() == "nan":
        return []

    return [item.strip() for item in value.split("|") if item.strip()]


def build_exchange_text(window_df: pd.DataFrame) -> str:
    parts = []

    for _, row in window_df.iterrows():
        step = row.get("conversation_step", "")
        current_text = str(row.get("current_text", "")).strip()
        paired_response_text = str(row.get("paired_response_text", "")).strip()

        if current_text:
            parts.append(f"[Exchange {step} - current] {current_text}")

        if paired_response_text:
            parts.append(f"[Exchange {step} - response] {paired_response_text}")

    return "\n".join(parts)


def get_mode_label(labels: list[str]) -> str:
    valid = [x for x in labels if x]
    if not valid:
        return ""

    mode_series = pd.Series(valid).mode()
    return mode_series.iloc[0] if not mode_series.empty else ""


def create_chunks(
    df: pd.DataFrame,
    window_size: int = 3,
    stride: int = 1,
) -> pd.DataFrame:
    required_cols = ["conversation_id", "conversation_step", "current_text", "paired_response_text"]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    work_df = df.copy()
    work_df["conversation_step"] = pd.to_numeric(work_df["conversation_step"], errors="coerce")
    work_df = work_df.sort_values(["conversation_id", "conversation_step"]).copy()

    all_chunks = []

    for conversation_id, group in work_df.groupby("conversation_id"):
        group = group.sort_values("conversation_step").reset_index(drop=True)

        n = len(group)
        if n == 0:
            continue

        if n < window_size:
            windows = [(0, n)]
        else:
            windows = []
            start = 0
            while start + window_size <= n:
                windows.append((start, start + window_size))
                start += stride

        for start_idx, end_idx in windows:
            window_df = group.iloc[start_idx:end_idx].copy()

            start_step = int(window_df["conversation_step"].iloc[0])
            end_step = int(window_df["conversation_step"].iloc[-1])

            labels = clean_list_values(window_df["label"].tolist()) if "label" in window_df.columns else []
            contexts = clean_list_values(window_df["context"].tolist()) if "context" in window_df.columns else []

            all_features = []
            if "features_clean_str" in window_df.columns:
                all_features = sorted(
                    set(
                        feature
                        for raw_value in window_df["features_clean_str"].tolist()
                        for feature in parse_pipe_separated_features(raw_value)
                    )
                )

            chunk_text = build_exchange_text(window_df)
            chunk_char_length = len(chunk_text)
            chunk_word_length = len(chunk_text.split())

            chunk_record = {
                "chunk_id": f"conv_{conversation_id}_steps_{start_step}_{end_step}",
                "conversation_id": conversation_id,
                "start_step": start_step,
                "end_step": end_step,
                "num_exchange_rows": len(window_df),
                "chunk_text": chunk_text,
                "chunk_label_mode": get_mode_label(labels),
                "chunk_final_label": labels[-1] if labels else "",
                "all_labels": labels,
                "all_labels_str": " | ".join(labels),
                "chunk_features": all_features,
                "chunk_features_str": " | ".join(all_features),
                "chunk_contexts": contexts,
                "chunk_contexts_str": " | ".join(contexts),
                "chunk_char_length": chunk_char_length,
                "chunk_word_length": chunk_word_length,
            }

            all_chunks.append(chunk_record)

    return pd.DataFrame(all_chunks)


def audit_chunks(chunks_df: pd.DataFrame) -> None:
    print("\n===== CHUNK AUDIT =====")
    print(f"Shape: {chunks_df.shape}")

    if len(chunks_df) == 0:
        print("No chunks created.")
        return

    print("\nChunk character length stats:")
    print(chunks_df["chunk_char_length"].describe())

    print("\nChunk word length stats:")
    print(chunks_df["chunk_word_length"].describe())

    if "chunk_final_label" in chunks_df.columns:
        print("\nChunk final label distribution:")
        print(chunks_df["chunk_final_label"].value_counts(dropna=False))

    print("\nSample chunks:")
    sample_cols = [
        "chunk_id",
        "conversation_id",
        "start_step",
        "end_step",
        "chunk_final_label",
        "chunk_features_str",
        "chunk_text",
    ]
    sample_cols = [col for col in sample_cols if col in chunks_df.columns]
    print(chunks_df[sample_cols].head(5))


def save_chunks(chunks_df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "chunks.csv"
    parquet_path = output_dir / "chunks.parquet"
    jsonl_path = output_dir / "chunks.jsonl"

    chunks_df.to_csv(csv_path, index=False)
    chunks_df.to_parquet(parquet_path, index=False)
    chunks_df.to_json(jsonl_path, orient="records", lines=True)

    print("\nSaved chunks to:")
    print(f" - {csv_path}")
    print(f" - {parquet_path}")
    print(f" - {jsonl_path}")