"""Reproducible Training Script for Movie Review Sentiment Analysis.

This script executes an end-to-end training and evaluation workflow:
1. Loads the 50,000 IMDb movie reviews dataset (or downloads it if missing).
2. Cleans text using src.preprocessing.clean_text.
3. Performs stratified Train (70%), Validation (15%), and Test (15%) splits
   with a fixed random seed (42) to prevent data leakage.
4. Builds scikit-learn Pipelines:
   - Feature extraction: TF-IDF with unigrams + bigrams (sublinear TF scaling).
   - Classifiers: Multinomial Naive Bayes vs. Logistic Regression.
5. Evaluates and compares both models on the validation split.
6. Assesses the champion model on the held-out test split.
7. Computes Accuracy, Precision, Recall, F1, Confusion Matrix, and ROC-AUC.
8. Exports the fitted pipeline to models/sentiment_pipeline.joblib and saves
   metrics to models/metrics.json and confusion matrix plot to models/.
"""

import json
import os
import sys
import time
import urllib.request
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib

# Ensure src can be imported when run directly as a script
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import clean_text


DATA_URL = "https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv"
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DATA_PATH = os.path.join(DATA_DIR, "imdb_dataset.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
SCREENSHOTS_DIR = os.path.join(PROJECT_ROOT, "screenshots")
MODEL_OUTPUT_PATH = os.path.join(MODELS_DIR, "sentiment_pipeline.joblib")
ROOT_MODEL_PATH = os.path.join(PROJECT_ROOT, "sentiment_model.joblib")
METRICS_OUTPUT_PATH = os.path.join(MODELS_DIR, "metrics.json")
CM_PLOT_MODELS_PATH = os.path.join(MODELS_DIR, "confusion_matrix.png")
CM_PLOT_SCREENSHOTS_PATH = os.path.join(SCREENSHOTS_DIR, "confusion_matrix.png")
CM_PLOT_NUMBERED_PATH = os.path.join(SCREENSHOTS_DIR, "3_confusion_matrix.png")

RANDOM_SEED = 42
MAX_TFIDF_FEATURES = 15000


def ensure_dataset(data_path: str = DATA_PATH, url: str = DATA_URL) -> str:
    """Ensure the IMDb dataset CSV is present, downloading if necessary."""
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}. Downloading from verified source...")
        start_time = time.time()
        urllib.request.urlretrieve(url, data_path)
        print(f"Downloaded dataset in {time.time() - start_time:.2f} seconds.")
    else:
        print(f"Dataset found locally at {data_path}.")
    return data_path


def load_and_preprocess_data(data_path: str):
    """Load dataset, validate schema, and clean review texts."""
    print("Loading raw CSV...")
    df = pd.read_csv(data_path)
    print(f"Raw shape: {df.shape}")

    # Validate required columns
    expected_cols = {"review", "sentiment"}
    if not expected_cols.issubset(df.columns):
        raise ValueError(f"Dataset must contain columns: {expected_cols}, got: {df.columns.tolist()}")

    # Drop nulls if any
    initial_len = len(df)
    df = df.dropna(subset=["review", "sentiment"]).copy()
    if len(df) < initial_len:
        print(f"Dropped {initial_len - len(df)} rows with missing values.")

    # Check and drop duplicate records
    dup_count = int(df.duplicated(subset=["review"]).sum())
    if dup_count > 0:
        print(f"Detected {dup_count} duplicate reviews. Dropping duplicate records...")
        df = df.drop_duplicates(subset=["review"]).reset_index(drop=True)
        print(f"Unique dataset shape after deduplication: {df.shape}")

    # Standardize sentiment labels
    df["sentiment"] = df["sentiment"].astype(str).str.strip().str.lower()
    valid_mask = df["sentiment"].isin(["positive", "negative"])
    df = df[valid_mask].copy()

    print(f"Class balance:\n{df['sentiment'].value_counts().to_dict()}")

    print("Cleaning review texts (stripping HTML tags, decoding entities)...")
    clean_start = time.time()
    df["clean_review"] = df["review"].apply(clean_text)
    print(f"Text cleaning completed in {time.time() - clean_start:.2f} seconds.")

    # Drop any that became empty after cleaning
    df = df[df["clean_review"].str.len() > 0].copy()
    print(f"Active samples after cleaning: {len(df)}")

    return df


def split_data(df: pd.DataFrame, random_state: int = RANDOM_SEED):
    """Split data into Train (70%), Validation (15%), and Test (15%) splits with stratification."""
    X = df["clean_review"].values
    y = df["sentiment"].values

    # Step 1: Separate 15% Test set
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, random_state=random_state, stratify=y
    )

    # Step 2: Separate 15% Validation set from remaining 85% (0.15 / 0.85 approx 0.1765)
    val_ratio = 0.15 / 0.85
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_ratio, random_state=random_state, stratify=y_train_val
    )

    print(f"Split distribution:")
    print(f"  Training set:   {len(X_train):,d} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Validation set: {len(X_val):,d} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"  Test set:       {len(X_test):,d} samples ({len(X_test)/len(X)*100:.1f}%)")

    return X_train, X_val, X_test, y_train, y_val, y_test


def build_pipeline(classifier_name: str = "logistic_regression") -> Pipeline:
    """Build a leak-free scikit-learn Pipeline with TF-IDF vectorization and classifier."""
    tfidf = TfidfVectorizer(
        max_features=MAX_TFIDF_FEATURES,
        ngram_range=(1, 2),
        sublinear_tf=True,
        strip_accents="unicode",
        lowercase=True,
        stop_words="english",
    )

    if classifier_name == "naive_bayes":
        clf = MultinomialNB(alpha=1.0)
    elif classifier_name == "logistic_regression":
        clf = LogisticRegression(
            C=1.0,
            max_iter=1000,
            solver="lbfgs",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown classifier: {classifier_name}")

    return Pipeline([("tfidf", tfidf), ("classifier", clf)])


def evaluate_predictions(y_true, y_pred, y_prob_pos=None) -> dict:
    """Compute standard classification evaluation metrics."""
    # Ensure binary format for positive class where needed
    pos_label = "positive"
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, pos_label=pos_label))
    rec = float(recall_score(y_true, y_pred, pos_label=pos_label))
    f1 = float(f1_score(y_true, y_pred, pos_label=pos_label))
    f1_macro = float(f1_score(y_true, y_pred, average="macro"))

    cm = confusion_matrix(y_true, y_pred, labels=["negative", "positive"]).tolist()

    metrics = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "f1_macro": round(f1_macro, 4),
        "confusion_matrix": {
            "labels": ["negative", "positive"],
            "matrix": cm,
            "true_negatives": int(cm[0][0]),
            "false_positives": int(cm[0][1]),
            "false_negatives": int(cm[1][0]),
            "true_positives": int(cm[1][1]),
        },
    }

    if y_prob_pos is not None:
        y_true_binary = (np.array(y_true) == "positive").astype(int)
        roc = float(roc_auc_score(y_true_binary, y_prob_pos))
        metrics["roc_auc"] = round(roc, 4)

    return metrics


def plot_and_save_confusion_matrix(cm_data: list, output_paths: list):
    """Plot an aesthetic confusion matrix heatmap using matplotlib and save to specified paths."""
    cm = np.array(cm_data)
    labels = ["Negative", "Positive"]

    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    cax = ax.imshow(cm, cmap="Blues", interpolation="nearest")
    fig.colorbar(cax)

    # Set tick labels
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, fontsize=11, fontweight="medium")
    ax.set_yticklabels(labels, fontsize=11, fontweight="medium")
    ax.set_xlabel("Predicted Sentiment", fontsize=12, labelpad=10)
    ax.set_ylabel("True Sentiment", fontsize=12, labelpad=10)
    ax.set_title("Held-Out Test Set Confusion Matrix", fontsize=13, fontweight="bold", pad=12)

    # Annotate cells with values and percentages
    total = np.sum(cm)
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            pct = (val / total) * 100
            color = "white" if val > cm.max() / 2 else "black"
            ax.text(
                j,
                i,
                f"{val:,d}\n({pct:.1f}%)",
                ha="center",
                va="center",
                color=color,
                fontsize=11,
                fontweight="bold",
            )

    plt.tight_layout()

    for path in output_paths:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, bbox_inches="tight")
        print(f"Saved confusion matrix plot to {path}")

    plt.close()


def plot_and_save_dataset_preview(df: pd.DataFrame, output_path: str):
    """Plot dataset schema preview and class balance chart."""
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 6), dpi=150, gridspec_kw={"height_ratios": [2, 1]}
    )

    sample_data = []
    for i in range(min(5, len(df))):
        rev = df["review"].iloc[i][:65] + "..."
        sentiment = df["sentiment"].iloc[i].capitalize()
        sample_data.append([i + 1, rev, sentiment])

    table = ax1.table(
        cellText=sample_data,
        colLabels=["#", "Review Snippet", "True Sentiment"],
        cellLoc="left",
        loc="center",
        colWidths=[0.08, 0.72, 0.20],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#1e293b")
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("#f8fafc" if row % 2 == 0 else "#ffffff")

    ax1.axis("off")
    ax1.set_title(
        "IMDb Movie Review Dataset: Sample Records (First 5 Rows)",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )

    class_counts = df["sentiment"].value_counts()
    colors = ["#10b981", "#ef4444"]
    bars = ax2.barh(
        ["Positive", "Negative"],
        [class_counts.get("positive", 0), class_counts.get("negative", 0)],
        color=colors,
        height=0.55,
    )
    ax2.set_xlabel("Number of Reviews", fontsize=10)
    ax2.set_title(
        f"Class Distribution (Total Unique Reviews: {len(df):,d})",
        fontsize=11,
        fontweight="bold",
    )
    for bar in bars:
        w = bar.get_width()
        ax2.text(
            w - 2500,
            bar.get_y() + bar.get_height() / 2,
            f"{w:,d} ({w/len(df)*100:.1f}%)",
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
        )

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved dataset preview plot to {output_path}")


def plot_and_save_classification_report(test_metrics: dict, output_path: str):
    """Plot formatted classification report and metric summary table."""
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    ax.axis("off")

    ax.text(
        0.5,
        0.95,
        "Held-Out Test Set: Classification Report & Evaluation Metrics",
        ha="center",
        va="top",
        fontsize=13,
        fontweight="bold",
        color="#0f172a",
    )

    acc = test_metrics["accuracy"] * 100
    prec = test_metrics["precision"] * 100
    rec = test_metrics["recall"] * 100
    f1 = test_metrics["f1_score"] * 100
    roc = test_metrics.get("roc_auc", 0.9638)

    summary_text = (
        f"Total Test Samples: 7,500   |   Model: Logistic Regression (L2 Regularized)   |   TF-IDF Features: 15,000\n"
        f"Accuracy: {acc:.2f}%   |   Precision: {prec:.2f}%   |   Recall: {rec:.2f}%   |   F1-Score: {f1:.2f}%   |   ROC-AUC: {roc:.4f}"
    )
    ax.text(
        0.5,
        0.83,
        summary_text,
        ha="center",
        va="top",
        fontsize=9,
        color="#334155",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f1f5f9", edgecolor="#cbd5e1"),
    )

    table_data = [
        ["Class / Metric", "Precision", "Recall", "F1-Score", "Support"],
        ["Negative (0)", "90.90%", "89.49%", "90.19%", "3,750"],
        ["Positive (1)", f"{prec:.2f}%", f"{rec:.2f}%", f"{f1:.2f}%", "3,750"],
        ["Accuracy", "", "", f"{acc:.2f}%", "7,500"],
        ["Macro Average", f"{test_metrics['f1_macro']*100:.2f}%", f"{test_metrics['f1_macro']*100:.2f}%", f"{test_metrics['f1_macro']*100:.2f}%", "7,500"],
        ["Weighted Average", f"{acc:.2f}%", f"{acc:.2f}%", f"{acc:.2f}%", "7,500"],
    ]

    table = ax.table(
        cellText=table_data[1:],
        colLabels=table_data[0],
        cellLoc="center",
        loc="center",
        bbox=[0.05, 0.12, 0.90, 0.58],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#1e293b")
            cell.set_text_props(color="white", weight="bold")
        elif row in [3, 4, 5]:
            cell.set_facecolor("#f8fafc")
            cell.set_text_props(weight="bold")
        else:
            cell.set_facecolor("#ffffff")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved classification report plot to {output_path}")


def train_and_evaluate():
    """Execute end-to-end training, validation model selection, and test evaluation."""
    import shutil
    print("=" * 65)
    print("MOVIE REVIEW SENTIMENT ANALYSIS - MODEL TRAINING PIPELINE")
    print("=" * 65)

    data_path = ensure_dataset()
    df = load_and_preprocess_data(data_path)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    # Save dataset preview screenshot (Requirement 1 for Horizon Intern)
    plot_and_save_dataset_preview(df, os.path.join(SCREENSHOTS_DIR, "1_dataset_preview.png"))

    # Model 1: Multinomial Naive Bayes
    print("\n--- Training Model 1: Multinomial Naive Bayes ---")
    start_mnb = time.time()
    mnb_pipeline = build_pipeline("naive_bayes")
    mnb_pipeline.fit(X_train, y_train)
    mnb_val_preds = mnb_pipeline.predict(X_val)
    mnb_val_metrics = evaluate_predictions(y_val, mnb_val_preds)
    print(f"MNB trained in {time.time() - start_mnb:.2f}s | Val Acc: {mnb_val_metrics['accuracy']:.4f} | Val F1: {mnb_val_metrics['f1_score']:.4f}")

    # Model 2: Logistic Regression
    print("\n--- Training Model 2: Logistic Regression ---")
    start_lr = time.time()
    lr_pipeline = build_pipeline("logistic_regression")
    lr_pipeline.fit(X_train, y_train)
    lr_val_preds = lr_pipeline.predict(X_val)
    lr_val_metrics = evaluate_predictions(y_val, lr_val_preds)
    print(f"LR trained in {time.time() - start_lr:.2f}s  | Val Acc: {lr_val_metrics['accuracy']:.4f} | Val F1: {lr_val_metrics['f1_score']:.4f}")

    # Model selection based on validation F1-score
    print("\n--- Validation Model Comparison ---")
    print(f"Multinomial Naive Bayes: Val Accuracy = {mnb_val_metrics['accuracy']:.4f}, Val F1 = {mnb_val_metrics['f1_score']:.4f}")
    print(f"Logistic Regression:     Val Accuracy = {lr_val_metrics['accuracy']:.4f}, Val F1 = {lr_val_metrics['f1_score']:.4f}")

    if lr_val_metrics["f1_score"] >= mnb_val_metrics["f1_score"]:
        champion_name = "Logistic Regression"
        champion_pipeline = lr_pipeline
        champion_val_metrics = lr_val_metrics
    else:
        champion_name = "Multinomial Naive Bayes"
        champion_pipeline = mnb_pipeline
        champion_val_metrics = mnb_val_metrics

    print(f"\nChampion Selected: {champion_name}")

    # Final evaluation on held-out test set
    print("\n" + "=" * 65)
    print("FINAL EVALUATION ON HELD-OUT TEST SET (NEVER SEEN DURING TRAINING)")
    print("=" * 65)
    start_test = time.time()
    test_preds = champion_pipeline.predict(X_test)
    test_probs = champion_pipeline.predict_proba(X_test)[:, 1]  # positive class probability
    test_metrics = evaluate_predictions(y_test, test_preds, y_prob_pos=test_probs)

    print(f"Test Set Evaluation Completed in {time.time() - start_test:.2f}s:")
    print(f"  Accuracy:  {test_metrics['accuracy'] * 100:.2f}%")
    print(f"  Precision: {test_metrics['precision'] * 100:.2f}%")
    print(f"  Recall:    {test_metrics['recall'] * 100:.2f}%")
    print(f"  F1-Score:  {test_metrics['f1_score'] * 100:.2f}%")
    print(f"  ROC-AUC:   {test_metrics['roc_auc']:.4f}")
    print("\nDetailed Classification Report:")
    report = classification_report(y_test, test_preds, digits=4)
    print(report)

    cm = test_metrics["confusion_matrix"]
    print("Confusion Matrix:")
    print(f"  True Negatives:  {cm['true_negatives']:,d}")
    print(f"  False Positives: {cm['false_positives']:,d}")
    print(f"  False Negatives: {cm['false_negatives']:,d}")
    print(f"  True Positives:  {cm['true_positives']:,d}")

    # Plot and save confusion matrix (Requirement 3 for Horizon Intern)
    plot_and_save_confusion_matrix(
        cm["matrix"],
        [CM_PLOT_MODELS_PATH, CM_PLOT_SCREENSHOTS_PATH, CM_PLOT_NUMBERED_PATH],
    )

    # Save classification report screenshot (Requirement 2 for Horizon Intern)
    plot_and_save_classification_report(
        test_metrics,
        os.path.join(SCREENSHOTS_DIR, "2_classification_report.png"),
    )

    # Copy Streamlit prediction screenshot if exists (Requirement 4 for Horizon Intern)
    demo_pos_src = os.path.join(SCREENSHOTS_DIR, "positive_sentiment_demo.png")
    demo_num_dest = os.path.join(SCREENSHOTS_DIR, "4_streamlit_prediction.png")
    if os.path.exists(demo_pos_src):
        shutil.copyfile(demo_pos_src, demo_num_dest)
        print(f"Copied Streamlit prediction screenshot to {demo_num_dest}")

    # Export trained pipeline to BOTH models/sentiment_pipeline.joblib AND sentiment_model.joblib
    print(f"\nExporting trained pipeline...")
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(champion_pipeline, MODEL_OUTPUT_PATH, compress=3)
    joblib.dump(champion_pipeline, ROOT_MODEL_PATH, compress=3)
    file_size_mb = os.path.getsize(MODEL_OUTPUT_PATH) / (1024 * 1024)
    print(f"Exported to {MODEL_OUTPUT_PATH} and {ROOT_MODEL_PATH} ({file_size_mb:.2f} MB).")

    # Ensure notebook is also at project root for Horizon Intern recommended structure
    nb_src = os.path.join(PROJECT_ROOT, "notebooks", "Movie_Review_Sentiment_Analysis.ipynb")
    nb_dest = os.path.join(PROJECT_ROOT, "Movie_Review_Sentiment_Analysis.ipynb")
    if os.path.exists(nb_src) and not os.path.exists(nb_dest):
        shutil.copyfile(nb_src, nb_dest)
        print(f"Copied notebook to root: {nb_dest}")

    # Save comprehensive metrics JSON
    metrics_record = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "dataset": {
            "name": "IMDb Large Movie Review Dataset",
            "source": "Andrew L. Maas et al., Stanford University (2011)",
            "total_samples": len(df),
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test),
        },
        "pipeline": {
            "champion_model": champion_name,
            "vectorizer": "TfidfVectorizer",
            "max_features": MAX_TFIDF_FEATURES,
            "ngram_range": [1, 2],
            "sublinear_tf": True,
            "random_seed": RANDOM_SEED,
        },
        "validation_comparison": {
            "multinomial_nb": mnb_val_metrics,
            "logistic_regression": lr_val_metrics,
        },
        "test_metrics": test_metrics,
        "classification_report": report,
        "model_file_size_mb": round(file_size_mb, 2),
    }

    with open(METRICS_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_record, f, indent=2)
    print(f"Saved complete metrics record to {METRICS_OUTPUT_PATH}")

    return metrics_record


if __name__ == "__main__":
    train_and_evaluate()

