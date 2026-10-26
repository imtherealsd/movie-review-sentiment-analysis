"""Movie Review Sentiment Analysis - Streamlit Web Application.

A responsive, portfolio-ready interactive NLP application for predicting movie
review sentiment using a scikit-learn TF-IDF + Logistic Regression pipeline.
"""

import json
import os
import sys
from typing import Dict, Optional

import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.inference import (
    DEFAULT_MODEL_PATH,
    load_sentiment_model,
    predict_sentiment,
)
from src.preprocessing import extract_text_diagnostics

METRICS_PATH = os.path.join(PROJECT_ROOT, "models", "metrics.json")

# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Movie Review Sentiment Analysis | ML Portfolio",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Custom Styling (CSS)
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Global Typography and Base Styling */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Hero Banner */
    .hero-container {
        padding: 1.5rem 1.8rem;
        border-radius: 12px;
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        color: #ffffff;
        margin-bottom: 1.5rem;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
        letter-spacing: -0.02em;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-bottom: 0.8rem;
    }
    .badge-pill {
        display: inline-block;
        padding: 0.22rem 0.65rem;
        font-size: 0.78rem;
        font-weight: 600;
        border-radius: 9999px;
        margin-right: 0.4rem;
        margin-bottom: 0.3rem;
    }
    .badge-primary { background-color: #3b82f6; color: #ffffff; }
    .badge-info { background-color: #0284c7; color: #ffffff; }
    .badge-dark { background-color: #334155; color: #e2e8f0; }

    /* Sentiment Result Cards */
    .result-card-positive {
        background: #f0fdf4;
        border: 2px solid #22c55e;
        border-radius: 12px;
        padding: 1.3rem;
        color: #14532d;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(34, 197, 94, 0.12);
    }
    .result-card-negative {
        background: #fef2f2;
        border: 2px solid #ef4444;
        border-radius: 12px;
        padding: 1.3rem;
        color: #7f1d1d;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.12);
    }
    .result-header {
        font-size: 1.6rem;
        font-weight: 700;
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin-bottom: 0.4rem;
    }

    /* Keyword Tags */
    .keyword-tag-pos {
        display: inline-block;
        background-color: #dcfce7;
        color: #15803d;
        border: 1px solid #86efac;
        padding: 0.2rem 0.55rem;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 500;
        margin: 0.2rem 0.3rem 0.2rem 0;
    }
    .keyword-tag-neg {
        display: inline-block;
        background-color: #fee2e2;
        color: #b91c1c;
        border: 1px solid #fca5a5;
        padding: 0.2rem 0.55rem;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 500;
        margin: 0.2rem 0.3rem 0.2rem 0;
    }

    /* Callout Note */
    .callout-box {
        background-color: #f8fafc;
        border-left: 4px solid #64748b;
        padding: 0.75rem 1rem;
        border-radius: 0 6px 6px 0;
        font-size: 0.85rem;
        color: #475569;
        margin-top: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Curated Example Reviews
# -----------------------------------------------------------------------------
EXAMPLE_REVIEWS = {
    "🌟 Masterpiece (Positive)": (
        "An absolute masterpiece of modern cinema! The directing was visionary, "
        "the screenplay was tight and emotionally gripping, and the lead actor gave an "
        "Oscar-worthy performance. Highly recommended to anyone who loves compelling storytelling."
    ),
    "💔 Total Flop (Negative)": (
        "A colossal waste of two hours. The acting was wooden, the plot was filled with "
        "glaring plot holes, and the special effects looked like a high-school project. "
        "Painfully boring and utterly devoid of charm."
    ),
    "🎭 Sarcastic / Mixed": (
        "If you enjoy listening to terrible dialogue while watching actors stare into the void "
        "for two hours, then this movie is an absolute triumph. Otherwise, stay far away."
    ),
    "⚖️ Nuanced Critique (Leaning Negative)": (
        "The cinematography was breathtaking and the musical score was quite memorable, "
        "but the script unfortunately falls apart completely in the third act. What could "
        "have been a classic ends up as a disappointing and shallow experience."
    ),
}

# -----------------------------------------------------------------------------
# Model Caching
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_cached_pipeline(model_path: Optional[str] = None):
    """Cache fitted pipeline across user sessions to ensure fast inference."""
    try:
        return load_sentiment_model(model_path), None
    except Exception as exc:
        return None, str(exc)


@st.cache_data
def get_model_metrics(metrics_path: str = METRICS_PATH) -> Optional[Dict]:
    """Load evaluation metrics JSON if available."""
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


# -----------------------------------------------------------------------------
# Application State Initialization
# -----------------------------------------------------------------------------
if "review_text" not in st.session_state:
    st.session_state["review_text"] = ""

def set_example_text(text: str):
    """Update review text state from example button."""
    st.session_state["review_text"] = text


# -----------------------------------------------------------------------------
# Sidebar: System Metadata, Model Specs & Documentation
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="padding: 0.75rem 0.9rem; background: linear-gradient(135deg, #1e293b, #0f172a); border-radius: 8px; border: 1px solid #334155; margin-bottom: 1rem;">
            <div style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: #38bdf8;">Production ML System</div>
            <div style="font-size: 1.12rem; font-weight: 700; color: #f8fafc; margin-top: 0.2rem;">NLP Sentiment Engine</div>
            <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 0.2rem;">IMDb 50k Benchmark Pipeline</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.subheader("Model Specifications")

    metrics_data = get_model_metrics()
    if metrics_data:
        test_m = metrics_data.get("test_metrics", {})
        st.markdown(
            f"""
            - **Model:** {metrics_data.get('pipeline', {}).get('champion_model', 'Logistic Regression')}
            - **Vectorizer:** TF-IDF (1–2 ngrams, sublinear)
            - **Max Features:** {metrics_data.get('pipeline', {}).get('max_features', 15000):,d}
            - **Dataset:** IMDb 50,000 Reviews (Stanford)
            - **Test Accuracy:** `{test_m.get('accuracy', 0.9027) * 100:.2f}%`
            - **Test F1-Score:** `{test_m.get('f1_score', 0.9034) * 100:.2f}%`
            - **Test ROC-AUC:** `{test_m.get('roc_auc', 0.9638):.4f}`
            """
        )
    else:
        st.markdown(
            """
            - **Model:** Logistic Regression (C=1.0)
            - **Vectorizer:** TF-IDF (1–2 ngrams)
            - **Features:** 15,000 max vocabulary
            - **Dataset:** IMDb 50k Reviews
            - **Reported Accuracy:** ~90.3%
            """
        )

    st.divider()

    st.subheader("Serialization Notice")
    st.info(
        "**Safe Model Loading:** Deserializing joblib/pickle files can run arbitrary code. "
        "This app strictly loads verified models generated internally by `src/train.py`.",
        icon="🛡️",
    )

    st.divider()
    st.subheader("System Architecture")
    st.caption(
        "**Production NLP Classification Pipeline.** "
        "Engineered with strict train-validation-test leakage prevention, "
        "TF-IDF n-gram feature extraction, and interpretable token contributions."
    )


# -----------------------------------------------------------------------------
# Main Application Header
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-container">
        <div class="hero-title">🎬 Movie Review Sentiment Analysis</div>
        <div class="hero-subtitle">
            Natural Language Processing system predicting sentiment polarity (Positive vs. Negative) from film critiques.
        </div>
        <div>
            <span class="badge-pill badge-primary">Scikit-Learn Pipeline</span>
            <span class="badge-pill badge-info">TF-IDF Vectorizer (n-grams)</span>
            <span class="badge-pill badge-dark">IMDb 50,000 Benchmark</span>
            <span class="badge-pill badge-dark">Leakage-Free Validation</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Pipeline Availability Check
# -----------------------------------------------------------------------------
pipeline, load_error = get_cached_pipeline()

if pipeline is None:
    st.error(
        f"**Trained Model Not Found or Failed to Load!**\n\n"
        f"Details: `{load_error}`\n\n"
        f"To train the model and generate `models/sentiment_pipeline.joblib`, "
        f"run the following command in your terminal:\n\n"
        f"```bash\npython src/train.py\n```\n\n"
        f"Or run the Google Colab notebook at `notebooks/Movie_Review_Sentiment_Analysis.ipynb`."
    )
    st.stop()


# -----------------------------------------------------------------------------
# Interactive Example Prompts
# -----------------------------------------------------------------------------
st.write("##### 📝 Select an Example Review (or write your own below):")
example_cols = st.columns(len(EXAMPLE_REVIEWS))

for col, (label, text) in zip(example_cols, EXAMPLE_REVIEWS.items()):
    with col:
        if st.button(label, use_container_width=True):
            set_example_text(text)

st.write("")

# -----------------------------------------------------------------------------
# Review Input Form
# -----------------------------------------------------------------------------
review_input = st.text_area(
    label="**Movie Review Text:**",
    value=st.session_state["review_text"],
    height=160,
    placeholder="Paste or type a movie review here (e.g., 'An outstanding film with magnificent acting and great pacing...')...",
    help="Enter an English movie review critique. HTML tags and excessive spacing will be automatically cleaned.",
)

# Dynamic text diagnostics
diag = extract_text_diagnostics(review_input)
col_stats1, col_stats2, col_stats3 = st.columns([1, 1, 3])
with col_stats1:
    st.caption(f"Character Count: **{diag['raw_char_count']}**")
with col_stats2:
    st.caption(f"Word Count: **{diag['word_count']}**")
with col_stats3:
    if diag["raw_char_count"] > 10000:
        st.caption("⚠️ *Long text detected; will be safely truncated for inference.*")

# Action Buttons
btn_col1, btn_col2, _ = st.columns([1.5, 1, 4])
with btn_col1:
    analyze_clicked = st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True)
with btn_col2:
    if st.button("🗑️ Clear", use_container_width=True):
        st.session_state["review_text"] = ""
        st.rerun()


# -----------------------------------------------------------------------------
# Inference Execution & Results
# -----------------------------------------------------------------------------
if analyze_clicked or review_input:
    if not review_input.strip():
        st.warning(
            "⚠️ **Please enter review text or select one of the examples above before analyzing.**",
            icon="⚠️",
        )
    else:
        try:
            with st.spinner("Classifying sentiment and extracting keyword signals..."):
                result = predict_sentiment(review_input, pipeline=pipeline)

            is_positive = result["sentiment"] == "Positive"
            sentiment_label = result["sentiment"]
            confidence_pct = result["confidence"] * 100
            prob_pos_pct = result["prob_positive"] * 100
            prob_neg_pct = result["prob_negative"] * 100

            # Verdict Banner Card
            if is_positive:
                st.markdown(
                    f"""
                    <div class="result-card-positive">
                        <div class="result-header">
                            <span>👍</span>
                            <span>{sentiment_label.upper()} SENTIMENT</span>
                        </div>
                        <div style="font-size: 1.05rem;">
                            Model affinity: <strong>{confidence_pct:.1f}%</strong> positive indication.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="result-card-negative">
                        <div class="result-header">
                            <span>👎</span>
                            <span>{sentiment_label.upper()} SENTIMENT</span>
                        </div>
                        <div style="font-size: 1.05rem;">
                            Model affinity: <strong>{confidence_pct:.1f}%</strong> negative indication.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Metric Cards & Probability Breakdown
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            with metric_col1:
                st.metric("Predicted Verdict", sentiment_label, delta=f"{confidence_pct:.1f}% confidence")
            with metric_col2:
                st.metric("Positive Probability", f"{prob_pos_pct:.1f}%")
            with metric_col3:
                st.metric("Negative Probability", f"{prob_neg_pct:.1f}%")

            # Probability Bars
            st.write("**Probability Distribution:**")
            st.progress(result["prob_positive"], text=f"Positive: {prob_pos_pct:.1f}% | Negative: {prob_neg_pct:.1f}%")

            # Technical Calibration Explanation
            st.markdown(
                f"""
                <div class="callout-box">
                    <strong>Confidence Interpretation:</strong> {result['confidence_disclaimer']}
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Keyword Interpretability
            pos_words = result["top_positive_features"]
            neg_words = result["top_negative_features"]

            if pos_words or neg_words:
                st.write("")
                st.subheader("Key Vocabulary Influencing This Decision")
                feat_col1, feat_col2 = st.columns(2)

                with feat_col1:
                    st.write("**Positive Signals Detected:**")
                    if pos_words:
                        html_tags = "".join(
                            f'<span class="keyword-tag-pos">+{w["token"]} (+{w["contribution"]:.2f})</span>'
                            for w in pos_words
                        )
                        st.markdown(html_tags, unsafe_allow_html=True)
                    else:
                        st.caption("No strong positive n-grams found in active vocabulary.")

                with feat_col2:
                    st.write("**Negative Signals Detected:**")
                    if neg_words:
                        html_tags = "".join(
                            f'<span class="keyword-tag-neg">-{w["token"]} ({w["contribution"]:.2f})</span>'
                            for w in neg_words
                        )
                        st.markdown(html_tags, unsafe_allow_html=True)
                    else:
                        st.caption("No strong negative n-grams found in active vocabulary.")

            # Diagnostics & Preprocessing Transparency Expander
            with st.expander("🔍 Inspect Preprocessed Text & Raw Diagnostics", expanded=False):
                st.markdown(f"**Cleaned Text (HTML Stripped & Normalized):**\n> {result['cleaned_text']}")
                st.json(result)

        except ValueError as val_err:
            st.error(f"Input validation error: {val_err}")
        except Exception as e:
            st.error(f"Inference error encountered: {str(e)}")

# -----------------------------------------------------------------------------
# Evaluation Performance & Confusion Matrix
# -----------------------------------------------------------------------------
st.write("")
with st.expander("📊 View Held-Out Test Set Performance & Confusion Matrix", expanded=False):
    cm_img_path = os.path.join(PROJECT_ROOT, "screenshots", "confusion_matrix.png")
    eval_col1, eval_col2 = st.columns([1.1, 1])
    with eval_col1:
        if os.path.exists(cm_img_path):
            st.image(
                cm_img_path,
                caption="Held-Out Test Set Confusion Matrix (7,438 samples, 15% stratified split)",
                use_container_width=True,
            )
    with eval_col2:
        st.markdown(
            """
            #### 🏆 Model Benchmark Results
            
            **Evaluated on 7,438 Unseen Reviews:**
            - **Accuracy:** `89.45%`
            - **Precision (Positive):** `88.43%`
            - **Recall (Positive):** `90.87%`
            - **F1-Score (Positive):** `89.63%`
            - **ROC-AUC:** `0.9623`
            - **True Negatives (TN):** `3,261`
            - **True Positives (TP):** `3,392`
            - **False Positives (FP):** `444`
            - **False Negatives (FN):** `341`

            ---

            #### ⚖️ Baseline Validation Comparison
            - **Logistic Regression (Champion):** `89.82%` val acc | `90.00%` val F1
            - **Multinomial Naive Bayes:** `86.78%` val acc | `87.05%` val F1
            - *Winner:* Logistic Regression with sublinear TF-IDF (1–2 ngrams).
            """
        )


# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.write("")
st.divider()
st.markdown(
    """
    <div style="text-align: center; color: #64748b; font-size: 0.85rem; padding: 1rem 0;">
        Movie Review Sentiment Analysis &bull; Production NLP Classification System &bull; Powered by Python, Scikit-Learn & Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)
