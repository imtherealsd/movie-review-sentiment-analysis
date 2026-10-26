"""Inference and Model Evaluation Module for Movie Review Sentiment Analysis.

This module provides safe model loading, input validation, inference execution,
probability calibration disclaimers, and local feature importance explanation
for scikit-learn TF-IDF + Logistic Regression pipelines.
"""

import os
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np

from src.preprocessing import clean_text, extract_text_diagnostics

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRIMARY_MODEL_PATH = os.path.join(ROOT_DIR, "sentiment_model.joblib")
FALLBACK_MODEL_PATH = os.path.join(ROOT_DIR, "models", "sentiment_pipeline.joblib")
DEFAULT_MODEL_PATH = PRIMARY_MODEL_PATH


class ModelSecurityWarning(UserWarning):
    """Warning emitted when loading serialized joblib models."""
    pass


def load_sentiment_model(model_path: Optional[str] = None):
    """Load a trained scikit-learn sentiment analysis pipeline.

    Resolves between primary root path (sentiment_model.joblib) and
    models/sentiment_pipeline.joblib.

    SECURITY NOTICE:
    Serialized joblib / pickle files can execute arbitrary code during
    unpickling. NEVER load model files from untrusted, unverified, or public
    user-uploaded sources. Only load models trained internally or verified via
    cryptographic hashes.

    Args:
        model_path: Optional custom path. If None, checks primary and fallback paths.

    Returns:
        Fitted scikit-learn Pipeline.

    Raises:
        FileNotFoundError: If the model file is missing at all resolved paths.
        ValueError: If file does not end with recognized model extensions (.joblib, .pkl).
    """
    if model_path is None:
        if os.path.exists(PRIMARY_MODEL_PATH):
            model_path = PRIMARY_MODEL_PATH
        elif os.path.exists(FALLBACK_MODEL_PATH):
            model_path = FALLBACK_MODEL_PATH
        else:
            model_path = PRIMARY_MODEL_PATH

    if not (model_path.endswith(".joblib") or model_path.endswith(".pkl")):
        raise ValueError(
            f"Invalid model file extension for '{model_path}'. Expected .joblib or .pkl."
        )

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at '{model_path}'. "
            f"Please run 'python src/train.py' or download the pre-trained pipeline."
        )

    pipeline = joblib.load(model_path)
    return pipeline


def extract_influential_keywords(
    pipeline: Any,
    cleaned_text: str,
    top_k: int = 5,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Extract top positive and negative vocabulary features present in the review.

    Computes token-level contribution: (TF-IDF value * Logistic Regression coefficient).
    This provides model interpretability without extra dependencies like SHAP/LIME.

    Args:
        pipeline: Fitted scikit-learn Pipeline with 'tfidf' and 'classifier' steps.
        cleaned_text: Preprocessed review text.
        top_k: Number of top features to return per sentiment direction.

    Returns:
        Tuple of (top_positive_features, top_negative_features).
    """
    try:
        tfidf = pipeline.named_steps.get("tfidf")
        clf = pipeline.named_steps.get("classifier")

        if tfidf is None or clf is None:
            return [], []

        # Vectorize single text
        X_vec = tfidf.transform([cleaned_text])
        feature_names = tfidf.get_feature_names_out()

        # Check if classifier has linear coefficients
        if not hasattr(clf, "coef_"):
            return [], []

        coefs = clf.coef_[0]

        # Find non-zero feature indices in this review
        _, non_zero_cols = X_vec.nonzero()
        contributions = []

        for col in non_zero_cols:
            word = feature_names[col]
            tfidf_val = X_vec[0, col]
            weight = coefs[col]
            score = float(tfidf_val * weight)
            contributions.append({
                "token": word,
                "weight": round(float(weight), 3),
                "contribution": round(float(score), 4),
            })

        # Sort contributions
        pos_contribs = sorted([c for c in contributions if c["contribution"] > 0], key=lambda x: x["contribution"], reverse=True)[:top_k]
        neg_contribs = sorted([c for c in contributions if c["contribution"] < 0], key=lambda x: x["contribution"])[:top_k]

        return pos_contribs, neg_contribs
    except Exception:
        # Fallback gracefully if model structure differs
        return [], []


def predict_sentiment(
    text: Any,
    pipeline: Optional[Any] = None,
    model_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Validate review text, execute inference, and return structured prediction diagnostics.

    Args:
        text: Raw user review text.
        pipeline: Pre-loaded scikit-learn Pipeline (if None, loaded from model_path).
        model_path: Path to serialized model if pipeline is not provided.

    Returns:
        Dictionary containing:
        - sentiment: 'Positive' or 'Negative'
        - confidence: float between 0.50 and 1.00
        - prob_positive: float between 0.00 and 1.00
        - prob_negative: float between 0.00 and 1.00
        - decision_score: float distance from decision boundary
        - raw_text: original input
        - cleaned_text: preprocessed text
        - diagnostics: word count, char count, etc.
        - top_positive_features: keywords influencing positive prediction
        - top_negative_features: keywords influencing negative prediction
        - confidence_disclaimer: technical note on probability calibration
    """
    # Text validation & diagnostics
    diagnostics = extract_text_diagnostics(text if isinstance(text, str) else "")
    cleaned = diagnostics["cleaned_preview"] if diagnostics["is_empty"] else clean_text(text)

    if diagnostics["is_empty"] or not cleaned:
        raise ValueError(
            "Input text is empty, contains only whitespace, or has no valid alphabetic content."
        )

    # Load model if not provided
    if pipeline is None:
        pipeline = load_sentiment_model(model_path)

    # Inference
    raw_pred = pipeline.predict([cleaned])[0]
    sentiment = "Positive" if str(raw_pred).lower() == "positive" else "Negative"

    # Probabilities
    prob_pos = 0.5
    prob_neg = 0.5
    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba([cleaned])[0]
        # In our pipeline: classes are ['negative', 'positive'] alphabetically
        classes = list(pipeline.classes_)
        if "positive" in classes:
            pos_idx = classes.index("positive")
            neg_idx = 1 - pos_idx
            prob_pos = float(probs[pos_idx])
            prob_neg = float(probs[neg_idx])
        else:
            prob_neg = float(probs[0])
            prob_pos = float(probs[1]) if len(probs) > 1 else 1.0 - prob_neg

    # Decision boundary distance if available
    decision_score = None
    if hasattr(pipeline, "decision_function"):
        try:
            decision_score = float(pipeline.decision_function([cleaned])[0])
        except Exception:
            decision_score = None

    confidence = max(prob_pos, prob_neg)

    # Extract keyword explanations
    pos_keywords, neg_keywords = extract_influential_keywords(pipeline, cleaned, top_k=5)

    disclaimer = (
        "Note: Confidence is computed from Logistic Regression's sigmoid output "
        "(1 / (1 + exp(-w^T x))). It reflects decision boundary margin rather than "
        "empirically calibrated posterior probability."
    )

    return {
        "sentiment": sentiment,
        "confidence": round(confidence, 4),
        "prob_positive": round(prob_pos, 4),
        "prob_negative": round(prob_neg, 4),
        "decision_score": round(decision_score, 4) if decision_score is not None else None,
        "raw_text": str(text),
        "cleaned_text": cleaned,
        "diagnostics": diagnostics,
        "top_positive_features": pos_keywords,
        "top_negative_features": neg_keywords,
        "confidence_disclaimer": disclaimer,
    }
