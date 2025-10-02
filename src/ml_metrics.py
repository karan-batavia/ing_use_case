# ml_metrics.py

import joblib
import json
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "sensitivity_classifier.joblib"
clf_bundle = joblib.load(MODEL_PATH)
classifier = clf_bundle["pipeline"]
labels = clf_bundle["labels"]

# Load your evaluation dataset (texts + true labels)
# For example purposes, imagine you have a CSV:
import pandas as pd
df = pd.read_csv("eval_data.csv")  # columns: "text", "true_label"

y_true = df["true_label"]
y_pred = classifier.predict(df["text"])

# Generate metrics
report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
matrix = confusion_matrix(y_true, y_pred).tolist()

# Save results
metrics = {
    "classification_report": report,
    "confusion_matrix": matrix,
    "labels": labels,
}

with open("ml_performance.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("Saved ML performance metrics to ml_performance.json")
