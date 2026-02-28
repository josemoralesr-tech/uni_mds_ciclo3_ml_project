from pathlib import Path
import json
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, recall_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

DATA_PATH = Path("data/training/credit_default_training.csv")
MODEL_PATH = Path("models/random_forest_credit_default.joblib")
FEATURES_PATH = Path("models/feature_names.json")
METRICS_PATH = Path("reports/metrics.json")

TARGET_COL = "default.payment.next.month"

def main():
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    if TARGET_COL not in df.columns:
        raise ValueError(f"No encuentro target '{TARGET_COL}'. Ejemplo columnas: {df.columns.tolist()[:15]}")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int)

    print("Splitting train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training RandomForest...")
    rf = RandomForestClassifier(
        n_estimators=400,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    rf.fit(X_train, y_train)

    print("Evaluating...")
    preds = rf.predict(X_test)
    proba = rf.predict_proba(X_test)[:, 1]

    metrics = {
        "f1": float(f1_score(y_test, preds)),
        "recall": float(recall_score(y_test, preds)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "report": classification_report(y_test, preds, output_dict=True),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

    print(classification_report(y_test, preds))
    print("F1:", metrics["f1"])
    print("Recall:", metrics["recall"])
    print("ROC-AUC:", metrics["roc_auc"])

    print("Saving artifacts...")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf, MODEL_PATH)

    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    FEATURES_PATH.write_text(json.dumps(list(X.columns), indent=2), encoding="utf-8")

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Saved model -> {MODEL_PATH}")
    print(f"Saved metrics -> {METRICS_PATH}")

if __name__ == "__main__":
    main()