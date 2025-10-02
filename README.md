**What’s in this repo**
-----------------------

*   Hybrid pipeline: TF-IDF (or hashing fallback) + rule features, with MODEL=logreg|rf. Optional transformer (MODEL=bert|roberta) if you install HF deps.
    
*   Your regex patterns + keyword groups drive features, heuristic labels, and redaction.
    
*   Replaces entities with placeholders like \[EMAIL\_001\], keeps a mapping + explanations.
    
*   Loads the trained model and classifies text (optionally prints probabilities).
    
*   **Streamlit app**: your app.py (login + main UI), plus helper src/classifier\_service.py (integration layer).
    

**1) Environment**
------------------

```bash
\# Create a virtual env (any manager is fine)

python3 -m venv .venv

source .venv/bin/activate

\# Core deps

pip install -U pip

pip install scikit-learn joblib streamlit

\# Optional: for transformers (BERT/RoBERTa)

pip install "transformers>=4.40" "datasets>=2.18" accelerate torch

\# Optional: for file extraction in Streamlit uploads

pip install python-docx beautifulsoup4 PyPDF2
```

**2) Train a model (baseline)**
-------------------------------

```bash
\# Point to your dataset (optional, default: raw\_prompts.txt)

export DATA\_PATH="raw\_prompts.txt"

\# Logistic Regression

MODEL=logreg python ml\_setup.py --train

\# or Random Forest

MODEL=rf python ml\_setup.py --train
```

Outputs:

*   sensitivity\_classifier.joblib (saved next to ml\_setup.py)
    
*   Console report + confusion matrix
    

> Troubleshooting:

*   **FileNotFoundError**: set DATA\_PATH=/path/to/raw\_prompts.txt
    
*   **empty vocabulary**: the script automatically falls back to a HashingVectorizer (no action needed).
    

### **(Optional) Train BERT/RoBERTa**

```bash

\# BERT

MODEL=bert python ml\_setup.py --train --data raw\_prompts.txt

\# RoBERTa

MODEL=roberta python ml\_setup.py --train --data raw\_prompts.txt

```

Outputs: hf\_sensitivity\_model/ (HF format) + labels.json.

**3) Predict quickly**
----------------------

```bash

\# After baseline training

python predict\_snippets.py --model ./sensitivity\_classifier.joblib \\

\--text "Policy reminder: AML screening guideline is deprecated." --proba

```

> If you hit AttributeError: Can't get attribute 'RuleFeatureizer' when loading the joblib:

*   The script **imports** ml\_setup to register the custom transformer class (already handled in our version of predict\_snippets.py).
    
*   If you renamed ml\_setup.py, update the import shim near the top of the predictor.
    

**4) Redact text with placeholders**
------------------------------------

```bash

\# Write outputs to current folder

python redact.py --in demo\_input.txt \\

\--out ./outputs/demo\_output.txt \\

\--map ./outputs/demo\_mapping.json \\

\--csv ./outputs/demo\_mapping.csv

```

*   demo\_output.txt: redacted text with \[TYPE\_###\] placeholders
    
*   demo\_mapping.json: per-line mapping (placeholder ↔ original, spans, explanation)
    
*   demo\_mapping.csv: flat table (one row per replacement)
    

> If you use a path like ./outputs/demo\_output.txt, make sure the folder exists (or add a small \_ensure\_parent() in \_write\_outputs to auto-create dirs).