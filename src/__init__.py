"""Movie Review Sentiment Analysis Source Package.

This package contains reusable preprocessing, model training, and inference
utilities for binary sentiment classification of movie reviews.
"""

from src.preprocessing import clean_text, clean_text_batch, extract_text_diagnostics
from src.inference import load_sentiment_model, predict_sentiment, extract_influential_keywords

__all__ = [
    "clean_text",
    "clean_text_batch",
    "extract_text_diagnostics",
    "load_sentiment_model",
    "predict_sentiment",
    "extract_influential_keywords",
]

