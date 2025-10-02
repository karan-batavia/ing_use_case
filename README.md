
# 

It uses:
- `regex.py` and `regex_queries.py` for rule-based features



---

## Expected JSON input format
Assumes a list in a `.json` file with objects like:
```json
{"id": "123", "text": "Some text here...", "label": 0}
```
- `id` (optional): document id
- `text` (required): raw text input
- `label` (optional): integer class label (for training). If missing, examples are used only for inference/redaction.

You can change field names with CLI flags.

---

## Install (Python >= 3.9)
```bash
pip install -r requirements.txt
```

> Note: The first run will download a Hugging Face RoBERTa model (`roberta-base` by default).

---

## Quickstart

### 1) Train
```bash
# Point to your dataset path
export DATA_PATH="/Users/haniehhajighasemi/Desktop/BECODE-2/ing_use_case/raw_prompts.txt"

# Train with Logistic Regression (or rf)
MODEL=logreg python3 ml_setup.py --train
```

### 2) Evaluate / Predict
```bash
python3 predict_snippets.py --model ./sensitivity_classifier.joblib \
  --file ./my_samples.txt
```
OR

```
python3 predict_snippets.py --model ./sensitivity_classifier.joblib \
  --text "Customer John Smith with national number 90.10.10-612-39 has account balance is 12,500 EUR."
```

### 3) Redact (with explanations + mapping)
```bash

python3 redact.py --in demo_input.txt \
                  --out ./outputs/demo_output.txt \
                  --map ./outputs/demo_mapping.json \
                  --csv ./outputs/demo_mapping.csv
```