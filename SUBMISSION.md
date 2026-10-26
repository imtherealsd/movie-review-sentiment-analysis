# 🎓 Horizon Machine Learning Internship — Final Project Submission

**Project Title**: Movie Review Sentiment Analysis: End-to-End NLP System  
**Track**: Machine Learning & Natural Language Processing (Project 2 of 2)  
**Complementary Project**: Project 1 (MNIST CNN Handwritten Digit Recognition)  
**Status**: Complete, Verified, Runnable & Evaluated  

---

## 📌 Executive Summary

This submission completes **Project 2 of 2** for the Horizon Machine Learning Internship. While Project 1 addressed computer vision (CNN image classification on MNIST), this project demonstrates mastery over **unstructured natural language processing (NLP)**, statistical feature representation, baseline model benchmarking, leak-free pipeline engineering, model interpretability, and interactive web application design.

The system is trained and evaluated on the benchmark **Stanford Large Movie Review Dataset (IMDb 50,000)** (Andrew L. Maas et al., ACL 2011). After removing 418 duplicate records, the model achieves **89.45% accuracy** and an **F1-score of 89.63%** on 7,438 held-out test reviews.

---

## 📂 Repository Deliverables Mapping

| Deliverable | File Path | Description |
| :--- | :--- | :--- |
| **Colab Notebook (Root)** | [`Movie_Review_Sentiment_Analysis.ipynb`](file:///d:/Horizon%20Internship/ML-II/Movie_Review_Sentiment_Analysis.ipynb) | Runnable in Google Colab with dataset download, deduplication, training, and export. |
| **Interactive Web App** | [`app.py`](file:///d:/Horizon%20Internship/ML-II/app.py) | Responsive Streamlit UI with example reviews, probability gauges, keyword contributions, and confusion matrix view. |
| **Trained Model Artifact** | [`sentiment_model.joblib`](file:///d:/Horizon%20Internship/ML-II/sentiment_model.joblib) | Primary fitted scikit-learn Pipeline (0.29 MB, trackable in Git, no LFS needed). |
| **Requirements Specification** | [`requirements.txt`](file:///d:/Horizon%20Internship/ML-II/requirements.txt) | Pinned, lightweight dependencies (`pandas`, `numpy`, `scikit-learn`, `joblib`, `matplotlib`, `streamlit`, `pytest`). |
| **Main Documentation** | [`README.md`](file:///d:/Horizon%20Internship/ML-II/README.md) | Technical documentation, video demo plan, and setup instructions. |
| **Required Screenshots** | [`screenshots/`](file:///d:/Horizon%20Internship/ML-II/screenshots/) | Contains all 4 required screenshots (`1_dataset_preview.png`, `2_classification_report.png`, `3_confusion_matrix.png`, `4_streamlit_prediction.png`). |
| **Text Preprocessing** | [`src/preprocessing.py`](file:///d:/Horizon%20Internship/ML-II/src/preprocessing.py) | HTML tag stripper, entity decoder, whitespace normalizer, and diagnostic metrics. |
| **Inference & Explainability** | [`src/inference.py`](file:///d:/Horizon%20Internship/ML-II/src/inference.py) | Dual-path model loader, probability breakdown, keyword impact extractor ($TF\text{-}IDF \times w_i$), and calibration warnings. |
| **Training Pipeline** | [`src/train.py`](file:///d:/Horizon%20Internship/ML-II/src/train.py) | Reproducible script for stratified splitting (70/15/15), baseline training, test evaluation, and screenshot generation. |
| **Unit Test Suite** | [`tests/`](file:///d:/Horizon%20Internship/ML-II/tests/) | 17 unit tests verifying text cleaning, pipeline loading, and prediction schemas (100% pass rate in <2s). |

---

## 🔬 Measured Empirical Results (Verified on Deduplicated Dataset)

> [!NOTE]
> All metrics below reflect genuine, reproducible measurements obtained from running `src/train.py` on the 49,582 unique IMDb reviews with `random_state=42`.

### 1. Validation Model Selection (7,438 Samples)
- **Multinomial Naive Bayes Baseline**: Validation Accuracy = `86.78%`, Validation F1 = `87.05%`
- **Logistic Regression (L2 Regularized)**: Validation Accuracy = `89.82%`, Validation F1 = `90.00%`
- **Decision**: Logistic Regression selected as champion (+3.04% accuracy gain).

### 2. Held-Out Test Set Performance (7,438 Samples)
- **Accuracy**: **`89.45%`**
- **Precision (Positive Class)**: **`88.43%`**
- **Recall (Positive Class)**: **`90.87%`**
- **F1-Score (Positive Class)**: **`89.63%`**
- **Macro F1-Score**: **`89.44%`**
- **ROC-AUC**: **`0.9623`**
- **True Negatives**: `3,261` | **True Positives**: `3,392`
- **False Positives**: `444` | **False Negatives**: `341`

---

## 🛡️ Key Machine Learning Engineering Standards

1. **Zero Data Leakage**:
   - Feature extraction (`TfidfVectorizer`) was fitted **strictly on the 34,706 training reviews** inside a scikit-learn `Pipeline`.
   - The test set was completely isolated until after model selection was finalized.
2. **Duplicate & Missing Value Handling**:
   - Explicitly checked and removed 418 duplicate records; verified zero null records.
3. **Probability Calibration Transparency**:
   - The interface explicitly informs users that raw logistic sigmoid scores $\sigma(w^T x + b)$ reflect decision margin rather than empirically calibrated posterior probabilities.
4. **Model Explainability**:
   - Token-level linear contributions are computed and displayed as positive and negative tags, demystifying the model's prediction for reviewers.
5. **Lightweight Deployment**:
   - Model artifact size is only **0.29 MB**, enabling instantaneous loading without Git LFS or external cloud buckets.
6. **No Published Secrets**:
   - System operates entirely offline; `.gitignore` strictly protects `.env`, cache, and sensitive files.

---

## ⚡ Reviewer Quickstart Commands

### 1. Local Application Execution
```powershell
pip install -r requirements.txt
streamlit run app.py
```
Open `http://localhost:8501`.

### 2. Run Test Suite
```powershell
python -m pytest -v
```
*(All 17 tests execute in <2 seconds.)*

### 3. Re-execute End-to-End Training (Optional)
```powershell
python src/train.py
```

---

## ✅ Final Submission Verification Checklist

- [x] Project structure matches all Horizon Intern specification requirements.
- [x] Colab notebook (`Movie_Review_Sentiment_Analysis.ipynb`) is present at root and runnable with deduplication and export.
- [x] Primary model artifact (`sentiment_model.joblib`) is saved at root and loads cleanly.
- [x] Strict train/validation/test split preventing data leakage.
- [x] Baseline comparison between Multinomial Naive Bayes and Logistic Regression.
- [x] Real empirical evaluation metrics recorded in `models/metrics.json` and plotted in `screenshots/3_confusion_matrix.png`.
- [x] All 4 required screenshots present in `screenshots/` (`1_dataset_preview.png`, `2_classification_report.png`, `3_confusion_matrix.png`, `4_streamlit_prediction.png`).
- [x] Interactive Streamlit app with review examples, diagnostics, keyword explainability, and error handling.
- [x] 17 automated unit tests covering preprocessing, loading, and inference.
- [x] 1–2 minute video demo plan included in `README.md`.
- [x] No unauthorized git pushes or public deployments performed without user approval.
