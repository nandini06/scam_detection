import pandas as pd
import re

from .config import STEP_MARKER_PATTERN
from .cleaners import clean_text_basic

def split_exchange_text(text: object) -> tuple[str, str]:
    """
    Split a row like:
    'Caller text ... [Step: 1] Assistant response ...'

    into:
    - current_text
    - paired_response_text

    If no [Step: n] marker is present, paired_response_text is empty.
    """
    text = clean_text_basic(text)

    if not text:
        return "", ""

    parts = re.split(STEP_MARKER_PATTERN, text, maxsplit=1)

    current_text = parts[0].strip()
    paired_response_text = parts[1].strip() if len(parts) > 1 else ""

    return current_text, paired_response_text


def has_step_marker(text: object) -> bool:
    if pd.isna(text):
        return False
    return bool(re.search(STEP_MARKER_PATTERN, str(text)))
