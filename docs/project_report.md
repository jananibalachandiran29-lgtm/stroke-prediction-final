# Project Report — Stroke Prediction Using Machine Learning

## Student Details
- **Name:** Janani B
- **Registration Number:** 25MIM10094
- **Program:** Integrated M.Tech AI Branch
- **Course Code:** CSA2001
- **Course:** Fundamentals in AI & ML
- **University:** VIT Bhopal University

## 1. Title
**Stroke Prediction Using Machine Learning**

## 2. Abstract
This project develops a supervised machine-learning system for binary stroke-risk classification. Patient demographic and health-related attributes are preprocessed and transformed into numerical features using DictVectorizer. A Logistic Regression classifier with balanced class weights is trained using an 80/20 stratified train-test split. The trained model is evaluated using accuracy, precision, recall, F1-score, ROC-AUC, and a confusion matrix. A command-line application allows a user to enter patient attributes and obtain a model-generated probability and class prediction. The system is intended as an academic demonstration rather than a clinical diagnostic tool.

## 3. Introduction
Machine learning can be used to identify patterns in structured healthcare datasets. In this project, a binary classification approach is applied to a stroke dataset. The work focuses on data preparation, feature encoding, model training, evaluation, serialization, and terminal-based inference.

## 4. Problem Statement
Develop a machine-learning model that predicts the binary `stroke` outcome from patient demographic and health-related attributes.

## 5. Objectives
1. Prepare the stroke dataset for machine learning.
2. Encode categorical attributes.
3. Train a Logistic Regression classifier.
4. Evaluate the classifier on unseen test data.
5. Save the model and feature vectorizer.
6. Provide a command-line prediction interface.

## 6. Dataset
The repository contains `stroke-data.csv` with 5,110 original records. The preprocessing stage removes records with missing values, leaving 4,909 usable rows for the reported experiment.

Main features: gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type, avg_glucose_level, bmi, smoking_status.

Target: stroke (0 or 1).

## 7. Methodology
Categorical and numerical attributes are prepared and transformed using `DictVectorizer`. An 80/20 stratified train-test split with `random_state=42` is used. Logistic Regression is trained with `max_iter=4000`, `class_weight=balanced`, `solver=lbfgs`, and `random_state=42`.

## 8. System Workflow
```text
Stroke Dataset → Data Cleaning → Feature Preparation → DictVectorizer
→ 80/20 Stratified Split → Logistic Regression → Evaluation
→ Saved Model → CLI Prediction
```

## 9. Experimental Results
Results from the actual execution of `python train.py`:

| Metric | Result |
|---|---:|
| Dataset rows after preprocessing | 4,909 |
| Training rows | 3,927 |
| Testing rows | 982 |
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

## 10. Implementation
- `train.py` loads data, trains/evaluates the model, and saves `model.pkl`.
- `main_app.py` loads the model/vectorizer and performs interactive command-line prediction.

## 11. Execution
```bash
pip install -r requirements.txt
python train.py
python main_app.py
```

## 12. Limitations
The model is dependent on the dataset and is not clinically validated. The target class is imbalanced, so multiple metrics are needed to interpret performance. Results from this academic model should not be treated as medical advice or a diagnosis.

## 13. Future Scope
- Cross-validation and hyperparameter tuning.
- Additional model comparisons.
- Improved class-imbalance strategies.
- Explainable AI techniques.
- Independent external validation.
- API deployment after further validation.

## 14. Conclusion
The project demonstrates an end-to-end supervised machine-learning workflow for binary classification, covering preprocessing, feature encoding, Logistic Regression, evaluation, model serialization, and terminal-based inference.

## 15. References
1. Scikit-learn documentation for Logistic Regression and model evaluation.
2. Pandas documentation for data loading and preprocessing.
3. The dataset included in the project repository (`stroke-data.csv`).
