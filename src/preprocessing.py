"""Text Preprocessing Module for Movie Review Sentiment Analysis.

This module provides reproducible, robust text cleaning functions designed
to handle real-world movie review data, including HTML artifacts, character
entities, and formatting irregularities.
"""

import html
import re
from typing import Any, List, Optional, Union


# Precompiled regex patterns for performance
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
WHITESPACE_PATTERN = re.compile(r"\s+")
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")

# Maximum safe text length for inference to prevent memory exhaustion / DoS
MAX_INFERENCE_CHARS = 50000


def clean_text(text: Any, max_length: Optional[int] = MAX_INFERENCE_CHARS) -> str:
    """Clean and normalize raw review text for sentiment classification.

    Steps applied:
    1. Type safety check (converts non-strings or None to empty string).
    2. Length truncation to safeguard against memory exhaustion.
    3. Unescape HTML entities (e.g. &quot; -> ", &amp; -> &).
    4. Strip HTML tags (e.g. <br />, <p>, etc., prevalent in IMDb reviews).
    5. Remove URLs.
    6. Normalize redundant whitespace, tabs, and newline characters.
    7. Strip leading and trailing whitespace.

    Note:
    We do not strip contractions or force lowercase here if downstream
    tokenizers (like TfidfVectorizer(lowercase=True)) handle them, but
    normalization here guarantees clean textual input.

    Args:
        text: Raw review input (string, None, or other type).
        max_length: Maximum allowed character length before truncation.

    Returns:
        Cleaned, normalized string.
    """
    if text is None:
        return ""

    if not isinstance(text, str):
        try:
            text = str(text)
        except Exception:
            return ""

    # Check for empty or whitespace-only
    if not text.strip():
        return ""

    # Truncate if excessively long
    if max_length is not None and len(text) > max_length:
        text = text[:max_length]

    # Decode HTML entities
    text = html.unescape(text)

    # Strip HTML tags
    text = HTML_TAG_PATTERN.sub(" ", text)

    # Strip URLs if present
    text = URL_PATTERN.sub(" ", text)

    # Collapse repeated whitespace
    text = WHITESPACE_PATTERN.sub(" ", text)

    return text.strip()


def clean_text_batch(texts: List[Any], max_length: Optional[int] = MAX_INFERENCE_CHARS) -> List[str]:
    """Apply clean_text across an iterable of texts.

    Args:
        texts: List or iterable of raw text items.
        max_length: Maximum allowed characters per text.

    Returns:
        List of cleaned strings.
    """
    return [clean_text(t, max_length=max_length) for t in texts]


def extract_text_diagnostics(raw_text: str) -> dict:
    """Extract lightweight diagnostic metrics from input text.

    Args:
        raw_text: The user-supplied or raw review text.

    Returns:
        Dictionary with character count, word count, line count, and cleaned preview.
    """
    cleaned = clean_text(raw_text)
    words = cleaned.split() if cleaned else []

    return {
        "raw_char_count": len(raw_text) if isinstance(raw_text, str) else 0,
        "clean_char_count": len(cleaned),
        "word_count": len(words),
        "cleaned_preview": (cleaned[:150] + "...") if len(cleaned) > 150 else cleaned,
        "is_empty": len(words) == 0,
    }
