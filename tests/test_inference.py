"""Unit Tests for Sentiment Inference and Model Pipeline."""

import os
import pytest
from src.inference import (
    DEFAULT_MODEL_PATH,
    extract_influential_keywords,
    load_sentiment_model,
    predict_sentiment,
)


def test_load_model_missing_file():
    """Verify that attempting to load a non-existent model raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_sentiment_model("models/non_existent_model.joblib")


def test_load_model_invalid_extension():
    """Verify that non-joblib/pkl files are rejected."""
    with pytest.raises(ValueError):
        load_sentiment_model("models/malicious_script.sh")


def test_load_trained_pipeline():
    """Verify that the genuine fitted pipeline loads and contains required steps."""
    pipeline = load_sentiment_model(DEFAULT_MODEL_PATH)
    assert pipeline is not None
    assert "tfidf" in pipeline.named_steps
    assert "classifier" in pipeline.named_steps
    assert hasattr(pipeline.named_steps["tfidf"], "vocabulary_")
    assert len(pipeline.named_steps["tfidf"].vocabulary_) > 1000

    # Test default zero-argument resolution
    auto_pipeline = load_sentiment_model()
    assert auto_pipeline is not None
    assert "classifier" in auto_pipeline.named_steps


def test_predict_sentiment_empty_input():
    """Verify that empty or whitespace-only inputs raise ValueError."""
    with pytest.raises(ValueError):
        predict_sentiment("")

    with pytest.raises(ValueError):
        predict_sentiment("    \n\t   ")


def test_predict_sentiment_positive_example():
    """Verify sentiment prediction and probability on an unmistakably positive review."""
    positive_review = (
        "An absolute masterpiece! The acting was superb, directing was visionary, "
        "and the emotional resonance stayed with me for days. One of the best films of the decade."
    )
    result = predict_sentiment(positive_review)

    assert result["sentiment"] == "Positive"
    assert result["prob_positive"] > 0.70
    assert result["prob_negative"] < 0.30
    assert result["confidence"] >= 0.70
    assert "confidence_disclaimer" in result
    assert result["diagnostics"]["word_count"] > 10
    assert len(result["top_positive_features"]) > 0


def test_predict_sentiment_negative_example():
    """Verify sentiment prediction and probability on an unmistakably negative review."""
    negative_review = (
        "Completely unwatchable garbage. Terrible acting, painfully boring plot, "
        "and horrendous pacing. A total waste of time and money."
    )
    result = predict_sentiment(negative_review)

    assert result["sentiment"] == "Negative"
    assert result["prob_negative"] > 0.70
    assert result["prob_positive"] < 0.30
    assert result["confidence"] >= 0.70
    assert len(result["top_negative_features"]) > 0


def test_predict_sentiment_html_embedded():
    """Verify that HTML tags inside user input are handled seamlessly during prediction."""
    html_review = "<div><b>Brilliant</b> cinema! Loved every second.<br /><br /></div>"
    result = predict_sentiment(html_review)
    assert result["sentiment"] == "Positive"
    assert "<br" not in result["cleaned_text"]


def test_keyword_influence_structure():
    """Verify that influential keywords return valid token, weight, and contribution fields."""
    pipeline = load_sentiment_model(DEFAULT_MODEL_PATH)
    text = "A truly wonderful performance with brilliant writing."
    pos_feats, neg_feats = extract_influential_keywords(pipeline, text, top_k=3)

    assert isinstance(pos_feats, list)
    if pos_feats:
        feat = pos_feats[0]
        assert "token" in feat
        assert "weight" in feat
        assert "contribution" in feat
        assert isinstance(feat["weight"], (int, float))
        assert feat["contribution"] > 0
