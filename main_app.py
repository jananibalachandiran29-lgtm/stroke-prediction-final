from pathlib import Path
import pickle
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model.pkl"

def load_artifact():
    try:
        with MODEL_PATH.open("rb") as f:
            a = pickle.load(f)
        if isinstance(a, dict) and "model" in a and "vectorizer" in a:
            return a["model"], a["vectorizer"]
    except (FileNotFoundError, EOFError, pickle.UnpicklingError, AttributeError):
        pass
    print("A compatible model was not found. Training the model now...\n")
    from train import train_model
    train_model()
    with MODEL_PATH.open("rb") as f:
        a = pickle.load(f)
    return a["model"], a["vectorizer"]

def ask_binary(prompt):
    while True:
        v = input(prompt).strip()
        if v in {"0", "1"}: return int(v)
        print("Please enter 0 or 1.")

def ask_number(prompt, minimum=0):
    while True:
        try:
            v = float(input(prompt).strip())
            if v >= minimum: return v
        except ValueError:
            pass
        print(f"Please enter a number >= {minimum}.")

def main():
    model, vectorizer = load_artifact()
    print("=" * 60)
    print("       AI/ML STROKE RISK PREDICTION SYSTEM")
    print("=" * 60)
    print("Academic project only — this is not a medical diagnosis.\n")
    patient = {
        "gender": input("Gender (Male/Female/Other): ").strip().lower().replace(" ", "_"),
        "age": ask_number("Age: "),
        "hypertension": {0: "absent", 1: "present"}[ask_binary("Hypertension (0 = No, 1 = Yes): ")],
        "heart_disease": {0: "absent", 1: "present"}[ask_binary("Heart disease (0 = No, 1 = Yes): ")],
        "ever_married": input("Ever married (Yes/No): ").strip().lower().replace(" ", "_"),
        "work_type": input("Work type (Private/Self-employed/Govt_job/children/Never_worked): ").strip().lower().replace(" ", "_"),
        "residence_type": input("Residence type (Urban/Rural): ").strip().lower().replace(" ", "_"),
        "avg_glucose_level": ask_number("Average glucose level: "),
        "bmi": ask_number("BMI: "),
        "smoking_status": input("Smoking status (formerly smoked/never smoked/smokes/Unknown): ").strip().lower().replace(" ", "_")
    }
    X = vectorizer.transform([patient])
    probability = float(model.predict_proba(X)[0, 1])
    prediction = int(model.predict(X)[0])
    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    print(f"Predicted stroke probability: {probability * 100:.2f}%")
    print(f"Predicted class: {prediction}")
    print("Class meaning: 0 = No stroke, 1 = Stroke")
    print("=" * 60)
    print("This output is for academic demonstration only and is not a diagnosis.")

if __name__ == "__main__":
    main()
