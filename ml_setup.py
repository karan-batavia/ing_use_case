#!/usr/bin/env python3
"""
Hybrid Sensitivity Classifier (rules + ML) with a simple model toggle.

- DATA_PATH can be set via env or --data argument
- MODEL can be: logreg | rf (roberta is a stub that raises NotImplementedError)
- Robust tokenization (unicode-friendly) + Hashing fallback to avoid 'empty vocabulary' errors
- Saves trained pipeline to sensitivity_classifier.joblib in the script directory
"""

import os
import re
import sys
import json
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
import joblib
from pathlib import Path

# --------------------------
# Config (env + sensible defaults)
# --------------------------
DEFAULT_DATA = "raw_prompts.txt"  # override via DATA_PATH or --data
DATA_PATH = os.environ.get("DATA_PATH", DEFAULT_DATA)
MODEL_CHOICE = os.environ.get("MODEL", "logreg").lower()   # logreg | rf | roberta(stub)
RANDOM_STATE = int(os.environ.get("SEED", "42"))
TEST_SIZE = float(os.environ.get("TEST_SIZE", "0.2"))
NGRAM_MAX = int(os.environ.get("NGRAM_MAX", "2"))

# Use integers for min_df to avoid over-pruning on small datasets
_env_min_df = os.environ.get("MIN_DF", "1")
MIN_DF = int(_env_min_df) if _env_min_df.isdigit() else 1
# Keep all tokens by default (no max_df pruning)
MAX_DF = float(os.environ.get("MAX_DF", "1.0"))

# --------------------------
# Optional user regex patterns
# --------------------------
USER_REGEX_OK = False
try:
    import regex as user_regex  
    USER_REGEX_OK = hasattr(user_regex, "PATTERNS")
except Exception:
    USER_REGEX_OK = False

FALLBACK_PATTERNS: Dict[str, str] = {
    "EMAIL": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
    "PHONE_EU": r"(?:\+\d{1,3}\s?)?(?:\d[\s-]?){9,}",
    "SSN_LIKE": r"\b\d{6}[- ]?\d{2,4}[\.]?\d{0,2}\b",
    "IBAN": r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b",
    "ACCOUNT_NUM": r"\b(?:acct|account)\s*\d{3,}\b",
    "AMOUNT": r"(?:USD|EUR|GBP|€|\$|£)\s?\d{1,3}(?:[, \u00A0]\d{3})*(?:\.\d{2})?",
    "DOB": r"\b\d{4}-\d{2}-\d{2}\b",
    "NATIONAL_ID": r"\bID[:\s-]?[A-Z0-9]{6,}\b",
    "BIOMETRIC": r"\b(FaceID|fingerprint|iris|biometric)\b",
}

def get_patterns() -> Dict[str, str]:
    if USER_REGEX_OK and isinstance(user_regex.PATTERNS, dict):
        return {**FALLBACK_PATTERNS, **user_regex.PATTERNS}  # user wins
    return FALLBACK_PATTERNS

PATTERNS = {k: re.compile(v, re.IGNORECASE) for k, v in get_patterns().items()}

# --------------------------
# Labels + heuristics
# --------------------------
LABELS = ["C1", "C2", "C3", "C4"]  # Public, Internal, Confidential, Highly Sensitive

def heuristic_label(text: str) -> str:
    t = text.lower()

    # C4: strong signals (PII/biometrics) or financial-personal specifics
    c4_hits = any(PATTERNS[k].search(text) for k in ["SSN_LIKE", "IBAN", "NATIONAL_ID", "BIOMETRIC"])
    email_hit = PATTERNS["EMAIL"].search(text) is not None
    amount_hit = PATTERNS["AMOUNT"].search(text) is not None
    c4_words = any(w in t for w in [
        "credit score","income","account balance","masked pin","national id","corpkeys","biometric",
        "sepa","wire transfer","cheque","direct debit"
    ])
    if c4_hits or (email_hit and amount_hit) or c4_words:
        return "C4"

    # C3: transactions, agreements, amounts (no direct PII)
    c3_words = any(w in t for w in [
        "agreement","supplier","customer","standing order","payment order","deposit","overdraft",
        "line of credit","bank guarantee","letter of credit","invoice","po-"
    ])
    if c3_words or amount_hit:
        return "C3"

    # C2: internal governance/policy/process
    c2_words = any(w in t for w in [
        "policy","guideline","standard","sop","governance","raci","owner","approver",
        "review cycle","deprecated","retired","under review","internal","digest","reminder","notice"
    ])
    if c2_words:
        return "C2"

    # C1: public docs/press/investor materials
    c1_words = any(w in t for w in [
        "annual report","pillar 3","press","newsroom","full year results","half year results",
        "public disclosures","investor","analyst","linkedin","press-room","pdf","xlsx","report"
    ])
    if c1_words:
        return "C1"

    # Default to internal
    return "C2"

# --------------------------
# Data loader
# --------------------------
def load_dataset(path: str) -> Tuple[List[str], List[str]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found at: {p}. "
                                f"Pass correct path via DATA_PATH env or --data argument.")
    texts: List[str] = []
    labels: List[str] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            texts.append(line)
            labels.append(heuristic_label(line))
    if not texts:
        raise ValueError("Dataset is empty after stripping blank lines.")
    return texts, labels

# --------------------------
# Rule-based featureizer
# --------------------------
class RuleFeatureizer(BaseEstimator, TransformerMixin):
    def __init__(self, pattern_keys: Optional[List[str]] = None):
        self.pattern_keys = pattern_keys or list(PATTERNS.keys())

    def fit(self, X: List[str], y=None):
        return self

    def transform(self, X: List[str]) -> Any:
        import numpy as np
        feats = []
        for text in X:
            row = []
            for k in self.pattern_keys:
                p = PATTERNS[k]
                matches = p.findall(text)
                row.append(int(bool(matches)))  # presence
                row.append(len(matches))        # count
            def has_any(words):
                tl = text.lower()
                return int(any(w in tl for w in words))
            row.extend([
                has_any(["credit score","income","account balance","masked pin","biometric"]),
                has_any(["agreement","supplier","customer","standing order","payment order","overdraft"]),
                has_any(["annual report","pillar 3","press","newsroom","full year results","investor"]),
                has_any(["policy","guideline","standard","sop","governance","raci","deprecated","retired"]),
            ])
            feats.append(row)
        return np.array(feats, dtype=float)

# --------------------------
# Feature builders
# --------------------------
def build_features(use_hash: bool = False) -> FeatureUnion:
    """
    Text features + rule features.
    - TF-IDF with unicode-friendly tokenization
    - Fallback to HashingVectorizer (no vocabulary step) if TF-IDF fails
    """
    rules = RuleFeatureizer()

    if use_hash:
        bow = HashingVectorizer(
            n_features=2**16,
            alternate_sign=False,
            lowercase=True,
            token_pattern=r"(?u)\b[\w'-]+\b",
            strip_accents='unicode',
        )
        return FeatureUnion([("bow", bow), ("rules", rules)])

    tfidf = TfidfVectorizer(
        ngram_range=(1, NGRAM_MAX),
        min_df=MIN_DF,
        max_df=MAX_DF,
        lowercase=True,
        token_pattern=r"(?u)\b[\w'-]+\b",  # keep words like “Smith’s”, “Q4-2024”
        strip_accents='unicode',
        stop_words=None,
    )
    return FeatureUnion([("tfidf", tfidf), ("rules", rules)])

# --------------------------
# Classifier chooser
# --------------------------
def build_classifier(name: str):
    if name == "logreg":
        # simple + reliable
        return LogisticRegression(max_iter=300, solver="lbfgs", multi_class="auto", random_state=RANDOM_STATE)
    if name == "rf":
        return RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, class_weight="balanced")
    if name == "roberta":
        raise NotImplementedError("RoBERTa fine-tune is not enabled in this simple setup. Use MODEL=logreg or MODEL=rf.")
    raise ValueError(f"Unknown MODEL: {name}")

# --------------------------
# Train / Eval
# --------------------------
@dataclass
class TrainResult:
    report: str
    conf_mat: List[List[int]]
    model_path: str

def train_and_eval(data_path: str) -> TrainResult:
    texts, labels = load_dataset(data_path)
    # Stratified split; if it fails (e.g., too few samples in a class), fall back to standard split
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=labels
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=None
        )

    clf = build_classifier(MODEL_CHOICE)

    # Try TF-IDF first, then fallback to hashing if vocabulary errors happen
    use_hash = False
    try:
        pipe = Pipeline([("features", build_features(use_hash)), ("clf", clf)])
        pipe.fit(X_train, y_train)
    except ValueError as e:
        msg = str(e).lower()
        if "empty vocabulary" in msg or "no terms remain" in msg:
            use_hash = True
            pipe = Pipeline([("features", build_features(use_hash)), ("clf", clf)])
            pipe.fit(X_train, y_train)
        else:
            raise

    y_pred = pipe.predict(X_test)
    report = classification_report(y_test, y_pred, digits=3, labels=LABELS, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=LABELS).tolist()

    out_path = str((Path(__file__).parent / "sensitivity_classifier.joblib").resolve())
    joblib.dump({"pipeline": pipe, "labels": LABELS}, out_path)

    return TrainResult(report=report, conf_mat=cm, model_path=out_path)

# --------------------------
# CLI
# --------------------------
def parse_args(argv: List[str]):
    data_arg = None
    if "--data" in argv:
        i = argv.index("--data")
        if i + 1 < len(argv):
            data_arg = argv[i + 1]
    return data_arg

def main(argv: List[str] = None):
    argv = argv or sys.argv[1:]
    data_arg = parse_args(argv)
    data_path = data_arg or DATA_PATH

    if "--train" in argv or len(argv) == 0:
        res = train_and_eval(data_path)
        print("=== Sensitivity Classifier (Hybrid: rules + TF-IDF + rules) ===")
        print(f"Model: {MODEL_CHOICE}")
        print(f"Data:  {data_path}")
        print("\n--- Classification Report ---")
        print(res.report)
        print("\n--- Confusion Matrix [rows=true, cols=pred] (order C1,C2,C3,C4) ---")
        print(json.dumps(res.conf_mat))
        print(f"\nModel saved to: {res.model_path}")
    else:
        print("Usage:")
        print("  DATA_PATH=/path/to/raw_prompts.txt MODEL=logreg python ml_setup.py --train")
        print("  python ml_setup.py --train --data /path/to/raw_prompts.txt")

if __name__ == "__main__":
    main()