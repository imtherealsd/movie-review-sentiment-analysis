"""Unit Tests for Text Preprocessing Module."""

import pytest
from src.preprocessing import clean_text, clean_text_batch, extract_text_diagnostics


def test_html_tag_stripping():
    """Verify that common HTML tags from web scrapers and IMDb reviews are removed."""
    raw = "The cinematography was breathtaking.<br /><br />However, the pacing dragged.<p>Overall a miss.</p>"
    cleaned = clean_text(raw)
    assert "<br" not in cleaned
    assert "</p>" not in cleaned
    assert "<p>" not in cleaned
    assert "The cinematography was breathtaking. However, the pacing dragged. Overall a miss." == cleaned


def test_html_entity_decoding():
    """Verify that HTML character entities are decoded to standard punctuation."""
    raw = "It&#39;s a &quot;masterpiece&quot; &amp; a must-watch!"
    cleaned = clean_text(raw)
    assert cleaned == "It's a \"masterpiece\" & a must-watch!"


def test_url_removal():
    """Verify that embedded links/URLs are stripped from review bodies."""
    raw = "Read full review at https://example.com/movie-review before watching."
    cleaned = clean_text(raw)
    assert "https://" not in cleaned
    assert "example.com" not in cleaned
    assert cleaned == "Read full review at before watching."


def test_whitespace_normalization():
    """Verify repeated whitespace, newlines, and tabs are collapsed."""
    raw = "   An     amazing\n\n\nexperience\t\twith great depth.   "
    cleaned = clean_text(raw)
    assert cleaned == "An amazing experience with great depth."


def test_empty_and_whitespace_only():
    """Verify empty and whitespace inputs safely return empty strings."""
    assert clean_text("") == ""
    assert clean_text("   \n\t   ") == ""
    assert clean_text(None) == ""


def test_non_string_types():
    """Verify handling of integers, floats, and objects."""
    assert clean_text(12345) == "12345"
    assert clean_text(3.14159) == "3.14159"


def test_max_length_truncation():
    """Verify that text exceeding max_length is safely truncated to avoid memory issues."""
    long_text = "word " * 20000  # 100,000 chars
    cleaned = clean_text(long_text, max_length=1000)
    assert len(cleaned) <= 1000


def test_clean_text_batch():
    """Verify batch processing across a list of texts."""
    samples = [
        "<b>Great!</b>",
        "Terrible<br />ending.",
        "",
        None,
    ]
    results = clean_text_batch(samples)
    assert results == ["Great!", "Terrible ending.", "", ""]


def test_extract_text_diagnostics():
    """Verify diagnostic metadata calculation."""
    raw = "<b>Classic film!</b> Highly recommended."
    diag = extract_text_diagnostics(raw)
    assert diag["raw_char_count"] == len(raw)
    assert diag["clean_char_count"] == len("Classic film! Highly recommended.")
    assert diag["word_count"] == 4
    assert diag["is_empty"] is False

    empty_diag = extract_text_diagnostics("   <br />   ")
    assert empty_diag["is_empty"] is True
    assert empty_diag["word_count"] == 0
