import pandas as pd

from .cleaners import (
    clean_list_values,
    parse_feature_list    
)

def reconstruct_conversations(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = ["conversation_id", "conversation_step", "current_text", "paired_response_text"]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(
                f"Cannot reconstruct conversations because '{col}' is missing."
            )

    work_df = df.copy()
    work_df["conversation_step"] = pd.to_numeric(work_df["conversation_step"], errors="coerce")
    work_df = work_df.sort_values(["conversation_id", "conversation_step"], na_position="last").copy()

    conversation_rows = []

    for conversation_id, group in work_df.groupby("conversation_id"):
        group = group.sort_values("conversation_step").copy()

        transcript_parts = []

        for _, row in group.iterrows():
            step = int(row["conversation_step"]) if pd.notna(row["conversation_step"]) else ""

            current_text = str(row["current_text"]).strip()
            paired_response_text = str(row["paired_response_text"]).strip()

            if current_text:
                transcript_parts.append(f"[Exchange {step} - current] {current_text}")

            if paired_response_text:
                transcript_parts.append(f"[Exchange {step} - response] {paired_response_text}")

        full_transcript = "\n".join(transcript_parts)

        # --- Context aggregation ---
        all_contexts = (
            clean_list_values(group["context"].tolist())
            if "context" in group.columns
            else []
        )
        context_joined = " | ".join(all_contexts)

        # --- Label aggregation ---
        valid_labels = [x for x in group["label"].tolist() if x]
        all_labels = valid_labels
        all_labels_str = " | ".join(all_labels)

        num_valid_labels = int((group["label"] != "").sum()) if "label" in group.columns else 0
        num_invalid_labels = int((group["label"] == "").sum()) if "label" in group.columns else 0

        conversation_label_mode = (
            pd.Series(valid_labels).mode().iloc[0]
            if valid_labels and not pd.Series(valid_labels).mode().empty
            else ""
        )
        conversation_final_label = valid_labels[-1] if valid_labels else ""

        # --- Feature aggregation ---
        all_features = []
        if "features_list" in group.columns:
            all_features = sorted(
                set(
                    feature
                    for feature_list in group["features_list"].tolist()
                    for feature in feature_list
                )
            )
        elif "features" in group.columns:
            all_features = sorted(
                set(
                    feature
                    for raw_value in group["features"].tolist()
                    for feature in parse_feature_list(raw_value)
                )
            )

        all_features_str = " | ".join(all_features)

        # --- Annotation aggregation ---
        all_annotations = (
            clean_list_values(group["annotations"].tolist())
            if "annotations" in group.columns
            else []
        )
        all_annotations_str = " | ".join(all_annotations)

        # --- Transcript stats ---
        transcript_char_length = len(full_transcript)
        transcript_word_length = len(full_transcript.split())

        conversation_rows.append({
            "conversation_id": conversation_id,
            "num_turn_rows": len(group),
            "max_conversation_step": group["conversation_step"].max(),
            "full_transcript": full_transcript,

            "context_list": all_contexts,
            "context_joined": context_joined,

            "conversation_label_mode": conversation_label_mode,
            "conversation_final_label": conversation_final_label,
            "num_valid_labels": num_valid_labels,
            "num_invalid_labels": num_invalid_labels,

            "all_labels": all_labels,
            "all_labels_str": all_labels_str,

            "all_features": all_features,
            "all_features_str": all_features_str,

            "all_annotations": all_annotations,
            "all_annotations_str": all_annotations_str,

            "transcript_char_length": transcript_char_length,
            "transcript_word_length": transcript_word_length,
        })

    return pd.DataFrame(conversation_rows)
