# 🎬 Movie Review Sentiment Analysis

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.7.2-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.56.0-FF4B4B.svg)](https://streamlit.io/)
[![Pytest](https://img.shields.io/badge/Pytest-9.0.2-green.svg)](https://pytest.org/)
[![Test Accuracy](https://img.shields.io/badge/Test%20Accuracy-89.45%25-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Horizon Intern — Machine Learning Internship Submission (Project 2)**  
> An end-to-end sentiment classification system that predicts whether a movie review is **Positive** or **Negative**, featuring leakage-free TF-IDF feature extraction, baseline model benchmarking, unit tests, and an interactive Streamlit web interface.

---

## 📑 Table of Contents
1. [Objective](#-objective)
2. [Technologies Used](#-technologies-used)
3. [Dataset Description & Source](#-dataset-description--source)
4. [Data Preprocessing Methodology](#-data-preprocessing-methodology)
5. [TF-IDF Feature Extraction](#-tf-idf-feature-extraction)
6. [Classification Algorithm](#-classification-algorithm)
7. [Training & Evaluation Methodology](#-training--evaluation-methodology)
8. [Actual Measured Evaluation Results](#-actual-measured-evaluation-results)
9. [Project Folder Structure](#-project-folder-structure)
10. [Screenshots Section](#-screenshots-section)
11. [Installation & Local Execution](#-installation--local-execution)
12. [Obtaining or Retraining the Model](#-obtaining-or-retraining-the-model)
13. [1–2 Minute Video Demo Plan](#-12-minute-video-demo-plan)
14. [Limitations](#-limitations)
15. [Author Information](#-author-information)

---

## 🎯 Objective

The objective of this project is to build an understandable, reproducible Natural Language Processing (NLP) system that classifies raw movie reviews into **Positive** or **Negative** sentiment. The project includes:
- Data acquisition, schema validation, duplicate removal, and text cleaning.
- Leakage-free TF-IDF feature extraction inside scikit-learn Pipelines.
- Benchmarking Multinomial Naive Bayes against Logistic Regression.
- Model serialization to `sentiment_model.joblib`.
- An interactive Streamlit web application (`app.py`) for live inference and review inspection.
- Automated unit tests (`tests/`) ensuring stability and correctness.

---

## 💻 Technologies Used

- **Programming Language**: Python 3.10
- **Machine Learning & NLP**: `scikit-learn` (TF-IDF Vectorizer, Logistic Regression, Multinomial Naive Bayes)
- **Data Manipulation**: `pandas`, `numpy`
- **Model Serialization**: `joblib`
- **Data Visualization**: `matplotlib`
- **Web Application Interface**: `streamlit`
- **Unit Testing**: `pytest`

---

## 📚 Dataset Description & Source

- **Dataset**: Stanford Large Movie Review Dataset (IMDb 50k)
- **Source & Citation**: Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. (2011). *"Learning Word Vectors for Sentiment Analysis"*. Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics (ACL 2011).
- **Total Records**: 50,000 raw movie reviews.
- **Deduplication**: 418 duplicate review texts were identified and removed, resulting in **49,582 unique reviews**.
- **Class Balance**: 24,884 Positive (50.2%) and 24,698 Negative (49.8%) — balanced binary classification.
- **License / Terms**: Freely available for academic, educational, and benchmarking use.

---

## 🧹 Data Preprocessing Methodology

Raw reviews from web datasets contain formatting artifacts that degrade model performance. The preprocessing pipeline applies:
1. **Handling Missing & Duplicate Data**: Verified zero null fields; dropped 418 duplicate records.
2. **HTML Tag Removal**: Strips tags such as `<br />`, `<p>`, and `<div>` using regular expressions.
3. **HTML Entity Unescaping**: Decodes character entities (e.g., `&quot;` $\rightarrow$ `"`, `&amp;` $\rightarrow$ `&`).
4. **URL Removal**: Strips web links (`http://...`, `https://...`).
5. **Whitespace Normalization**: Collapses repeated spaces, tabs, and newlines into single spaces.
6. **Input Validation**: Safe handling of empty text, non-string types, and truncation protection for text exceeding 50,000 characters.

Both offline training (`src/train.py`) and live inference (`src/inference.py`, `app.py`) share the exact same `clean_text()` implementation in `src/preprocessing.py`.

---

## 🔍 TF-IDF Feature Extraction

Term Frequency-Inverse Document Frequency (TF-IDF) converts unstructured text strings into informative numerical vectors:

$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \log\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1$$

### Configuration Parameters:
- **`max_features=15000`**: Retains the top 15,000 most informative n-grams, keeping model size compact (<0.3 MB).
- **`ngram_range=(1, 2)`**: Extracts unigrams (`"good"`, `"bad"`) and bigrams (`"not good"`, `"waste time"`), capturing local negation and compound phrases.
- **`sublinear_tf=True`**: Replaces raw term frequency with $1 + \log(\text{tf})$ to prevent repetitive words from dominating.
- **`stop_words='english'`**: Removes non-informative English function words.
- **`lowercase=True`**: Standardizes casing.

---

## ⚖️ Classification Algorithm

We benchmarked two standard text classification algorithms:

1. **Multinomial Naive Bayes (`MultinomialNB(alpha=1.0)`)**: Fast probabilistic baseline assuming conditional independence between features given the class label.
2. **Logistic Regression (`LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs')`)**: Linear decision boundary model optimizing cross-entropy loss with L2 regularization.

**Outcome**: Logistic Regression outperformed Multinomial Naive Bayes on the validation set (+3.04% accuracy), and was selected as the champion model.

---

## 🛡️ Training & Evaluation Methodology

### Preventing Data Leakage
- The 49,582 unique reviews were split into:
  - **70% Training Set (34,706 reviews)**
  - **15% Validation Set (7,438 reviews)**
  - **15% Held-Out Test Set (7,438 reviews)**
- Partitioning used `train_test_split` with `stratify=y` and a fixed random seed (`random_state=42`).
- **Zero Leakage**: Feature vectorization is fitted **strictly on the 34,706 training reviews** inside a scikit-learn `Pipeline`. The test set is transformed only and remained untouched until final evaluation.

---

## 📊 Actual Measured Evaluation Results

> [!NOTE]
> All figures below are actual, verified numbers from running `python src/train.py` on the deduplicated IMDb dataset.

### 1. Validation Set Model Comparison (7,438 samples)

| Model Architecture | Training Time | Validation Accuracy | Validation F1-Score | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Multinomial Naive Bayes** | 17.49s | 86.78% | 87.05% | Baseline |
| **Logistic Regression (L2)** | 17.85s | **89.82%** | **90.00%** | 🏆 **Champion Selected** |

### 2. Final Evaluation on Held-Out Test Set (7,438 samples)

| Evaluation Metric | Measured Value | Description |
| :--- | :---: | :--- |
| **Accuracy** | **89.45%** | Correct predictions / total test samples |
| **Precision (Positive)** | **88.43%** | True positives / total predicted positive |
| **Recall (Positive)** | **90.87%** | True positives / actual positive samples |
| **F1-Score (Positive)** | **89.63%** | Harmonic mean of precision & recall |
| **Macro F1-Score** | **89.44%** | Unweighted average F1 across both classes |
| **ROC-AUC Score** | **0.9623** | Area under the ROC curve |

#### Detailed Classification Report:
```
              precision    recall  f1-score   support

    negative     0.9053    0.8802    0.8926      3705
    positive     0.8843    0.9087    0.8963      3733

    accuracy                         0.8945      7438
   macro avg     0.8948    0.8944    0.8944      7438
weighted avg     0.8948    0.8945    0.8944      7438
```

#### Held-Out Confusion Matrix:
- **True Negatives (TN)**: 3,261
- **False Positives (FP)**: 444
- **False Negatives (FN)**: 341
- **True Positives (TP)**: 3,392

---

## 📁 Project Folder Structure

```
movie-review-sentiment-analysis/
├── Movie_Review_Sentiment_Analysis.ipynb   # Colab notebook at root
├── app.py                                 # Interactive Streamlit application
├── sentiment_model.joblib                 # Trained pipeline (0.29 MB)
├── requirements.txt                       # Project dependencies
├── README.md                              # Main documentation
│
├── screenshots/                           # Required submission visual assets
│   ├── 1_dataset_preview.png              # Dataset schema & class distribution
│   ├── 2_classification_report.png        # Classification report & metrics
│   ├── 3_confusion_matrix.png             # Test set confusion matrix heatmap
│   ├── 4_streamlit_prediction.png         # Working Streamlit app prediction
│   └── README.md                          # Visual asset walkthrough
│
├── src/                                   # Source modules
│   ├── __init__.py                        # Package init
│   ├── preprocessing.py                   # Text cleaning & diagnostics
│   ├── inference.py                       # Prediction & keyword influence
│   └── train.py                           # Training & screenshot generator
│
├── models/                                # Model artifacts
│   ├── sentiment_pipeline.joblib          # Pipeline copy
│   ├── metrics.json                       # Measured evaluation JSON
│   └── confusion_matrix.png               # Confusion matrix image
│
├── notebooks/                             # Secondary notebook path
│   └── Movie_Review_Sentiment_Analysis.ipynb
│
└── tests/                                 # Unit tests
    ├── __init__.py
    ├── test_preprocessing.py              # Text cleaning test cases
    └── test_inference.py                  # Model loading & inference test cases
```

---

## 📸 Screenshots Section

All four required visual deliverables are present in the [`screenshots/`](screenshots/) directory:

### 1. Dataset Preview (`screenshots/1_dataset_preview.png`)
![Dataset Preview](screenshots/1_dataset_preview.png)

### 2. Model Evaluation & Classification Report (`screenshots/2_classification_report.png`)
![Classification Report](screenshots/2_classification_report.png)

### 3. Confusion Matrix (`screenshots/3_confusion_matrix.png`)
![Confusion Matrix](screenshots/3_confusion_matrix.png)

### 4. Working Streamlit Application Showing Prediction (`screenshots/4_streamlit_prediction.png`)
![Streamlit Prediction](screenshots/4_streamlit_prediction.png)

---

## 🛠️ Installation & Local Execution

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/movie-review-sentiment-analysis.git
cd movie-review-sentiment-analysis
```

### 2. Set Up Virtual Environment & Install Dependencies
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# macOS / Linux:
source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### 3. Launch the Streamlit Web Application
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your web browser.

### 4. Run Unit Tests
```bash
python -m pytest -v
```
*(All 17 unit tests execute in <2 seconds without requiring model retraining.)*

---

## 🔄 Obtaining or Retraining the Model

### Option A: Use the Included Model Artifact
The repository already includes the trained, compact pipeline (`sentiment_model.joblib`, 0.29 MB). You can immediately run `streamlit run app.py` or run tests without retraining.

### Option B: Retrain Locally via Python Script
```bash
python src/train.py
```
This script will automatically:
1. Verify/download the IMDb dataset into `data/`.
2. Clean text and drop duplicate records.
3. Train MultinomialNB and Logistic Regression on the stratified training set.
4. Evaluate both on the validation set, select the champion, and evaluate on the test set.
5. Export `sentiment_model.joblib` and refresh all evaluation metrics and screenshot plots.

### Option C: Retrain in Google Colab
1. Open [`Movie_Review_Sentiment_Analysis.ipynb`](Movie_Review_Sentiment_Analysis.ipynb) in [Google Colab](https://colab.research.google.com/).
2. Run all cells (**Runtime > Run all**).
3. The final cell triggers a download of the trained `sentiment_model.joblib` directly to your local computer.

---

## 🎙️ 1–2 Minute Video Demo Plan

Use this outline when recording your internship submission video:

| Timestamp | Screen View | Spoken Content |
| :--- | :--- | :--- |
| **0:00 – 0:15** | Streamlit App Homepage | *"Hello. I am presenting Project 2 of my Machine Learning Internship: Movie Review Sentiment Analysis. The objective is to classify unstructured movie critiques as positive or negative using scikit-learn, TF-IDF n-grams, and Streamlit."* |
| **0:15 – 0:35** | `1_dataset_preview.png` & Preprocessing | *"We used the benchmark IMDb 50k dataset, removed 418 duplicates, and cleaned HTML tags and entities. We then used a leakage-free 70/15/15 stratified train-val-test split, fitting TF-IDF strictly on the training partition."* |
| **0:35 – 0:55** | `2_classification_report.png` & Confusion Matrix | *"We benchmarked Multinomial Naive Bayes against Logistic Regression. Logistic Regression was selected as the champion, achieving 89.45% accuracy and an F1-score of 89.63% on 7,438 unseen test reviews."* |
| **0:55 – 1:20** | Live App: Positive Review Test | *"In the Streamlit app, I click '🌟 Masterpiece'. Preprocessing cleans the text, and the model classifies it as POSITIVE with 95.5% confidence, highlighting key positive tokens like 'brilliant', 'superb', and 'masterpiece'."* |
| **1:20 – 1:45** | Live App: Negative Review Test | *"Next, I select a negative review. It is predicted as NEGATIVE with 99.6% confidence, driven by words like 'waste', 'dreadful', and 'painful'. A technical callout explains that confidence scores represent decision margins rather than calibrated probabilities."* |
| **1:45 – 2:00** | Terminal (`pytest -v`) & Conclusion | *"All 17 unit tests pass in under 2 seconds. The model artifact is compact at 0.29 MB, and the project is fully reproducible via Google Colab and local Python. Thank you."* |

---

## ⚠️ Limitations

1. **Contextual Sarcasm**: Bag-of-words and n-gram representations struggle with sarcastic sentences (e.g., *"If you enjoy watching paint dry, this is a masterpiece"*).
2. **Probability Calibration**: Raw logistic sigmoid outputs $\sigma(w^T x + b)$ reflect decision boundary margins rather than strictly calibrated empirical probabilities.
3. **Long-Distance Dependencies**: Bigram windows cannot resolve negations separated by more than two words (e.g., *"I could never, in any circumstance, recommend this"*).

---

## 👤 Author Information

- **Internship**: Horizon Intern — Machine Learning Track
- **Project**: Movie Review Sentiment Analysis (Project 2 of 2)
- **Environment**: Python 3.10, Scikit-Learn 1.7.2, Streamlit 1.56.0
