import pandas as pd
from .config import VALID_LABELS

def clean_text_basic(text: object) -> str:
    if pd.isna(text):
        return ""
    text = str(text)
    text = text.replace("\n", " ").replace("\r", " ")
    text = " ".join(text.split())
    return text.strip()


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


def parse_feature_list(value: object) -> list[str]:
    if pd.isna(value):
        return []

    value = str(value).strip()
    if not value or value.lower() == "nan":
        return []

    parts = [item.strip() for item in value.split(",")]
    parts = [normalize_feature_name(item) for item in parts if item.strip()]
    return [item for item in parts if item and item.lower() != "nan"]


def normalize_label(value: object) -> str:
    if pd.isna(value):
        return ""

    text = str(value).strip().lower().strip('"').strip("'")
    text = " ".join(text.split())

    # map common variants
    replacements = {
        " scam": "scam",
        " neutral": "safe",
        " legitimate": "safe",
        "scam ": "scam",
        "neutral ": "safe",
        "legitimate ": "safe",
        "scam response": "scam",
        "potential scam": "suspicious",
        "slightly suspicious": "scam",
        "highly suspicious": "scam",
        "slightly_suspicious": "scam",
        "hightly_suspicious": "scam",
        "scam": "scam",
        "neutral": "safe",
        "legitimate": "safe",
        "standard_opening, identification_request": "safe",
        "identification_request": "safe",
        "standard_opening": "safe",
        "polite_ending": "safe",
        "potential_scam": "scam",
        " scam_response": "scam",
        " dismissing official protocols\"": "scam",
        " citing urgency\"": "scam",
        " adhering to protocols\"": "safe",
        "Scam": "scam",
        " suggesting a dangerous situation\"": "scam",
        " emphasizing security and compliance\"": "safe",
        " ready for further engagement\"": "scam",
        "suspicious": "scam",
        "uncertain": "safe",
    }

    text = replacements.get(text, text)

    return text if text in VALID_LABELS else "safe"


def normalize_feature_name(feature: str) -> str:
    feature = str(feature).strip().lower()
    feature = feature.replace(" ", "_")
    return feature
