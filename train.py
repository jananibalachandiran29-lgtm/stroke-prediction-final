from pathlib import Path
import pickle
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "stroke-data.csv"
MODEL_PATH = ROOT / "model.pkl"

CATEGORICAL = ["gender", "hypertension", "heart_disease", "ever_married", "work_type", "residence_type", "smoking_status"]
NUMERICAL = ["age", "avg_glucose_level", "bmi"]

def prepare_data():
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.lower().str.replace(" ", "_", regex=False)
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.lower().str.replace(" ", "_", regex=False)
    df["hypertension"] = df["hypertension"].map({0: "absent", 1: "present"})
    df["heart_disease"] = df["heart_disease"].map({0: "absent", 1: "present"})
    df = df.dropna().reset_index(drop=True)
    return df[CATEGORICAL + NUMERICAL], df["stroke"].astype(int)

def train_model():
    X, y = prepare_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
    vectorizer = DictVectorizer(sparse=False)
    X_train_vec = vectorizer.fit_transform(X_train.to_dict(orient="records"))
    X_test_vec = vectorizer.transform(X_test.to_dict(orient="records"))
    model = LogisticRegression(max_iter=4000, class_weight="balanced", solver="lbfgs", random_state=42)
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)
    y_prob = model.predict_proba(X_test_vec)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "report": classification_report(y_test, y_pred, zero_division=0),
        "rows": len(X), "train_rows": len(X_train), "test_rows": len(X_test)
    }
    with MODEL_PATH.open("wb") as f:
        pickle.dump({"model": model, "vectorizer": vectorizer}, f)
    return metrics

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    m = train_model()
    print("=" * 60)
    print("STROKE PREDICTION MODEL TRAINING")
    print("=" * 60)
    print(f"Dataset rows : {m['rows']}")
    print(f"Training rows: {m['train_rows']}")
    print(f"Testing rows : {m['test_rows']}")
    print(f"Accuracy     : {m['accuracy']:.4f}")
    print(f"Precision    : {m['precision']:.4f}")
    print(f"Recall       : {m['recall']:.4f}")
    print(f"F1-score     : {m['f1']:.4f}")
    print(f"ROC-AUC      : {m['roc_auc']:.4f}")
    print("\nConfusion Matrix:")
    print(m["confusion_matrix"])
    print("\nClassification Report:")
    print(m["report"])
    print(f"Model saved to: {MODEL_PATH}")

if __name__ == "__main__":
    main()
