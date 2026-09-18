# Stroke Prediction Using Machine Learning

An end-to-end machine learning pipeline for pre-processing clinical data, building predictive models, and evaluating stroke risk.

## Overview

This repository contains a modular Python application designed to predict patient stroke risk based on demographic and clinical attributes.

## 1. Project Overview

This project is a simple command-line machine learning application for stroke-risk classification. I used patient demographic and health-related information as input and trained a Logistic Regression model to predict the stroke class.

The main steps I followed are:

**Dataset → Preprocessing → Feature Encoding → Train/Test Split → Model Training → Evaluation → Prediction**

> **Academic disclaimer:** This project is for educational purposes only. It is not a medical diagnostic system and should not be used to make clinical decisions.

---

## 2. Problem Statement

The objective is to build a supervised machine-learning model that learns from historical patient records and predicts the binary target variable `stroke` from available patient attributes.

- `0` → No stroke
- `1` → Stroke

---

## 3. Objectives

1. Load and prepare a stroke dataset.
2. Handle missing values and categorical variables.
3. Convert categorical attributes into numerical features.
4. Train a Logistic Regression classifier.
5. Evaluate the classifier using standard classification metrics.
6. Save the trained model and feature vectorizer.
7. Provide a terminal-based prediction interface.

---

## 4. Dataset

The dataset used in this project is `stroke-data.csv`. The dataset has 5,110 records before rows with missing values are removed during preprocessing.

Features used by the model:

| Feature | Type | Description |
|---|---|---|
| `gender` | Categorical | Patient gender |
| `age` | Numerical | Patient age |
| `hypertension` | Binary | Hypertension indicator |
| `heart_disease` | Binary | Heart disease indicator |
| `ever_married` | Categorical | Marriage status |
| `work_type` | Categorical | Employment category |
| `Residence_type` | Categorical | Urban or rural residence |
| `avg_glucose_level` | Numerical | Average glucose level |
| `bmi` | Numerical | Body Mass Index |
| `smoking_status` | Categorical | Smoking category |
| `stroke` | Target | Binary stroke outcome |

Rows containing missing values are removed before training. In this dataset, this leaves 4,909 usable rows for the reported experiment.

---

## 5. Technologies Used

- Python 3.9+
- Pandas
- Scikit-learn
- Pickle
- Command Prompt / PowerShell / Terminal

---

## 6. Machine Learning Methodology

### Step 1: Data Loading

The CSV dataset is loaded using Pandas.

### Step 2: Data Preprocessing

- Column names are standardized.
- Text categories are normalized to lowercase and underscores.
- `hypertension` and `heart_disease` are represented as categorical values (`present` / `absent`).
- Rows with missing values are removed.
- The `stroke` column is separated as the target.

### Step 3: Feature Encoding

`DictVectorizer` converts the categorical and numerical feature dictionaries into a numerical feature matrix.

### Step 4: Train/Test Split

The data is divided using:

- **80% training data**
- **20% testing data**
- `random_state=42`
- Stratified sampling based on the target class

### Step 5: Model Training

The classifier is Logistic Regression with:

```text
max_iter = 4000
class_weight = balanced
solver = lbfgs
random_state = 42
```

`class_weight="balanced"` is used because the target classes are imbalanced.

### Step 6: Evaluation

The test set is evaluated using accuracy, precision, recall, F1-score, ROC-AUC, confusion matrix, and a classification report.

---

## 7. Experimental Results

The following results were obtained by running `python train.py` on the repository dataset with the fixed 80/20 stratified split and `random_state=42`.

| Metric | Test-set result |
|---|---:|
| Accuracy | 0.7566 |
| Precision | 0.1137 |
| Recall | 0.6905 |
| F1-score | 0.1953 |
| ROC-AUC | 0.8102 |

### Confusion Matrix

```text
[[714, 226],
 [ 13,  29]]
```

The positive-class results should be interpreted together with the class imbalance rather than using accuracy alone.

---

## 8. Repository Structure

```text
stroke-prediction/
│
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── stroke-data.csv
├── train.py
├── main_app.py
└── docs/
    └── project_report.md
```

`model.pkl` is generated locally by `train.py` and is intentionally not required in the repository. The prediction application automatically trains the model if the local model file is missing.

---

## 10. Requirements

Install **Python 3.9 or later** and Git.

Install dependencies from the repository root:

```bash
pip install -r requirements.txt
```

---

## 11. Environment Setup

Clone the repository:

```bash
git clone https://github.com/jananibalachandiran29-lgtm/stroke-prediction.git
```

Enter the project directory:

```bash
cd stroke-prediction
```

Optional but recommended: create a virtual environment.

### Windows

```bash
python -m venv venv
venv\\Scripts\\activate
```

### Linux/macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 12. Train and Evaluate the Model

From the repository root, run:

```bash
python train.py
```

The program will:

1. Load `stroke-data.csv`.
2. Preprocess the data.
3. Split the dataset into training and testing sets.
4. Train Logistic Regression.
5. Print evaluation metrics.
6. Save the trained artifact as `model.pkl`.

---

## 13. Run the Command-Line Prediction Application

After training, run:

```bash
python main_app.py
```

If a compatible `model.pkl` is not present, the application automatically trains the model first.

The program asks for:

- Gender
- Age
- Hypertension
- Heart disease
- Ever-married status
- Work type
- Residence type
- Average glucose level
- BMI
- Smoking status

It then prints the predicted probability and binary class.

Example output format:

```text
============================================================
RESULT
============================================================
Predicted stroke probability: XX.XX%
Predicted class: 0 or 1
Class meaning: 0 = No stroke, 1 = Stroke
============================================================
```

---

## 14. Reproducibility

The experiment is reproducible using the fixed `random_state=42` and stratified 80/20 train-test split.

To reproduce the reported results:

```bash
pip install -r requirements.txt
python train.py
```

The displayed metrics are generated from the actual test split at runtime.

---

## 15. Limitations

- The model is dependent on the dataset used for training.
- The target variable is imbalanced, so accuracy alone is not sufficient to describe performance.
- Results can change when a different dataset or split is used.
- The model is not clinically validated.
- The output is a machine-learning prediction, not a medical diagnosis.

---

## 16. Future Scope

Possible extensions include:

- Cross-validation and systematic hyperparameter tuning.
- Comparison with additional classification algorithms.
- More advanced imbalance-handling techniques.
- Feature selection and explainability methods.
- External validation on an independent dataset.
- API-based deployment after validating the core model.

---

## 17. Learning Outcomes

This project demonstrates practical use of:

- Supervised machine learning
- Binary classification
- Data preprocessing
- Categorical feature encoding
- Train/test splitting
- Logistic Regression
- Class imbalance handling
- Classification metrics
- Model serialization
- Command-line execution

---

## 18. Project Report

The project report is available at:

```text
docs/project_report.md
```

It contains the project description, methodology, experimental results, limitations, future scope, and conclusion.

---

