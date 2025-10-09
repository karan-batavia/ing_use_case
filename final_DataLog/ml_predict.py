#!/usr/bin/env python3
"""
Unified ML + Deep Learning Sensitivity Classifier with Auto Model Selection
--------------------------------------------------------------------------
Supports:
  - Classical ML (LogReg, RF)
  - Deep Neural Network (10-layer)
  - Auto-selection of best model based on metrics
  - Confidence prediction and hybrid (ensemble) mode
  - Integrated scrubbing (DataFog + Presidio)
"""

import os
import json
import joblib
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple, Optional
import random

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

from datafog import DataFog
from presidio_analyzer import AnalyzerEngine

LABELS = ["C1", "C2", "C3", "C4"]
RANDOM_STATE = 42
MODEL_PATHS = {
    "logreg": "logreg_model.joblib",
    "rf": "rf_model.joblib",
    "deepnn": "deep_model.keras",
}
META_PATH = "vectorizer_meta.joblib"
SELECTION_PATH = "model_selection.joblib"
STOP_FLAG = "training.stop"

# ---------------------------
# Heuristic labeling
# ---------------------------
def heuristic_label(text: str, strictness: str = "medium") -> str:
    """
    Heuristically assign a sensitivity label (C1..C4) to free text.

    This function uses a lightweight keyword-scoring approach instead of
    brittle single-keyword checks. It is intentionally fast and deterministic
    so it can be used to weakly label training data when no gold labels
    exist (as required by the project deliverable).

    Parameters
    - text: input string
    - strictness: one of 'low', 'medium', 'high' controlling thresholds.

    Returns
    - One of 'C1', 'C2', 'C3', 'C4'

    Notes
    - Keep the default behavior similar to the previous implementation by
      using 'medium' strictness.
    - This is purposely simple; if you want probabilistic scores, use
      `label_scores(text)` below.
    """
    t = (text or "").lower()

    # keyword buckets with relative weights
    KEYWORDS = {
        "C4": ["iban", "ssn", "social security", "credit card", "credit", "card number", "biometric", "passport", "cvv"],
        "C3": ["agreement", "invoice", "supplier", "payment", "transaction", "bank transfer"],
        "C2": ["policy", "internal", "guideline", "standard", "confidential", "customer", "client"],
        "C1": ["press", "public", "report", "investor", "announcement", "release"],
    }

    # strictness adjusts how many hits are needed to select a higher class
    STRICTNESS_THRESHOLDS = {
        "low": 0.8,
        "medium": 1.0,
        "high": 1.5,
    }
    thresh = STRICTNESS_THRESHOLDS.get(strictness, 1.0)

    # score each class by summing keyword matches (simple count, weighted)
    scores = {k: 0.0 for k in KEYWORDS}
    for label, kws in KEYWORDS.items():
        for kw in kws:
            if kw in t:
                # weight exact phrase matches slightly higher
                scores[label] += 1.0

    # normalize by keyword list length to avoid bias toward classes with more keywords
    norm_scores = {}
    for k, v in scores.items():
        denom = max(1, len(KEYWORDS[k]))
        norm_scores[k] = v / denom

    # final decision: pick the class with the highest normalized score, but
    # apply thresholding so weak matches fall back to C2 (default)
    best_label = max(norm_scores, key=norm_scores.get)
    best_score = norm_scores[best_label]

    if best_score < (0.05 * thresh):
        # almost no signal -> default to C2
        return "C2"
    return best_label


def label_scores(text: str) -> Dict[str, float]:
    """Return normalized keyword hit-scores for each class (helper for debugging).

    This function mirrors the heuristic used in `heuristic_label` and is
    useful for logging, thresholds tuning and explainability in audits.
    """
    t = (text or "").lower()
    KEYWORDS = {
        "C4": ["iban", "ssn", "social security", "credit card", "credit", "card number", "biometric", "passport", "cvv"],
        "C3": ["agreement", "invoice", "supplier", "payment", "transaction", "bank transfer"],
        "C2": ["policy", "internal", "guideline", "standard", "confidential", "customer", "client"],
        "C1": ["press", "public", "report", "investor", "announcement", "release"],
    }
    scores = {}
    for label, kws in KEYWORDS.items():
        count = 0
        for kw in kws:
            if kw in t:
                count += 1
        scores[label] = count / max(1, len(kws))
    return scores

# ---------------------------
# Scrubbing utilities
# ---------------------------
class DynamicScrubber:
    def __init__(self):
        self.dfog = DataFog()
        self.pengine = AnalyzerEngine()

    def scrub(self, text: str):
        # Be defensive: DataFog API may differ across versions. We'll prefer
        # generating deterministic placeholders ourselves using Presidio's
        # AnalyzerEngine so downstream descrubbing is reliable.
        try:
            entities = self.pengine.analyze(text=text, language="en")
        except Exception:
            entities = []

        if not entities:
            # Try to use DataFog if it exposes a simple anonymize-like method
            for method_name in ("clean_text", "anonymize", "anonymize_text", "scrub_text", "scrub", "mask"):
                fn = getattr(self.dfog, method_name, None)
                if callable(fn):
                    try:
                        result = fn(text)
                        # If result is a string, return it with an empty placeholder map
                        if isinstance(result, str):
                            return result, {}
                    except Exception:
                        continue
            # Nothing to anonymize — return original text
            return text, {}

        # Build placeholder mapping and replace spans in the original text.
        # Sort entities by start so replacements don't shift indices.
        entities_sorted = sorted(entities, key=lambda e: e.start)
        placeholder_map = {}
        out_parts = []
        last_idx = 0
        counts = {}
        for e in entities_sorted:
            # Skip entities that overlap with previously handled spans
            if e.start < last_idx:
                continue
            # append text before entity
            out_parts.append(text[last_idx : e.start])
            etype = e.entity_type if hasattr(e, 'entity_type') else getattr(e, 'type', 'ENTITY')
            counts.setdefault(etype, 0)
            counts[etype] += 1
            ph = f"<{etype}_{counts[etype]}>"
            out_parts.append(ph)
            placeholder_map[ph] = text[e.start : e.end]
            last_idx = e.end

        out_parts.append(text[last_idx:])
        scrubbed_text = "".join(out_parts)
        return scrubbed_text, placeholder_map

    def descrub(self, text: str, placeholder_map: Dict[str, List[str]]) -> str:
        # placeholder_map may map to a single string or a list; support both.
        for ph, vals in placeholder_map.items():
            repl = None
            if isinstance(vals, (list, tuple)) and vals:
                repl = vals[0]
            elif isinstance(vals, str):
                repl = vals
            if repl is not None:
                text = text.replace(ph, repl)
        return text

# ---------------------------
# Deep model
# ---------------------------
def build_deep_model(input_dim: int, num_classes: int = 4):
    model = Sequential([
        Dense(512, activation='relu', input_dim=input_dim),
        BatchNormalization(), Dropout(0.3),
        Dense(256, activation='relu'), BatchNormalization(), Dropout(0.3),
        Dense(128, activation='relu'), BatchNormalization(), Dropout(0.3),
        Dense(64, activation='relu'), BatchNormalization(), Dropout(0.2),
        Dense(32, activation='relu'), BatchNormalization(),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# ---------------------------
# Classical model
# ---------------------------
def build_pipeline(model_type="logreg"):
    if model_type == "logreg":
        clf = LogisticRegression(max_iter=400, random_state=RANDOM_STATE)
    elif model_type == "rf":
        clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
    else:
        raise ValueError("Invalid model type")
    # use a correct word-boundary token pattern (single backslashes)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b")
    return Pipeline([("tfidf", vectorizer), ("clf", clf)])

@dataclass
class TrainResult:
    model_type: str
    metrics: Dict[str, float]
    conf_mat: List[List[int]]
    model_path: str

# ---------------------------
# Training
# ---------------------------
def train_model(data_dir: str, model_type="logreg", deep_epochs: int = 10, sample_per_class: Optional[int] = None) -> TrainResult:
    texts, labels = [], []
    files = list(Path(data_dir).glob("*.txt"))
    for f in files:
        txt = f.read_text(encoding="utf-8", errors="ignore")
        texts.append(txt)
        labels.append(heuristic_label(txt))

    # Optional per-class sampling to limit dataset size (useful for deep models)
    if sample_per_class is not None:
        by_label: Dict[str, List[str]] = {l: [] for l in LABELS}
        for t, l in zip(texts, labels):
            if l in by_label:
                by_label[l].append(t)
        sampled_texts, sampled_labels = [], []
        for l, items in by_label.items():
            if len(items) > sample_per_class:
                items = random.sample(items, sample_per_class)
            for it in items:
                sampled_texts.append(it)
                sampled_labels.append(l)
        texts, labels = sampled_texts, sampled_labels

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=RANDOM_STATE, stratify=labels
    )

    # quick check for an external stop flag so long-running training can be
    # interrupted cleanly by the UI creating the STOP_FLAG file
    try:
        if Path(STOP_FLAG).exists():
            print("Stop flag detected before training; aborting train_model early...")
            raise KeyboardInterrupt("Training aborted via stop flag")
    except Exception:
        # In case of any filesystem oddness, continue with training
        pass

    if model_type == "deepnn":
        vectorizer = TfidfVectorizer(max_features=8000)
        X_train_vec = vectorizer.fit_transform(X_train).toarray()
        X_test_vec = vectorizer.transform(X_test).toarray()
        y_train_idx = np.array([LABELS.index(y) for y in y_train])
        y_test_idx = np.array([LABELS.index(y) for y in y_test])
        y_train_oh = to_categorical(y_train_idx, len(LABELS))
        y_test_oh = to_categorical(y_test_idx, len(LABELS))

        model = build_deep_model(X_train_vec.shape[1])
        # Add a callback that checks for the STOP_FLAG file between epochs
        try:
            from tensorflow.keras.callbacks import Callback

            class StopOnFile(Callback):
                def on_epoch_end(self, epoch, logs=None):
                    try:
                        if Path(STOP_FLAG).exists():
                            print(f"Stop flag detected during deep model training at epoch {epoch}; stopping training.")
                            self.model.stop_training = True
                    except Exception:
                        pass

            callbacks = [StopOnFile()]
        except Exception:
            callbacks = []

        model.fit(X_train_vec, y_train_oh, validation_split=0.1, epochs=deep_epochs, batch_size=32, verbose=1, callbacks=callbacks)

        probs = model.predict(X_test_vec)
        preds = [LABELS[int(np.argmax(p))] for p in probs]
        acc = accuracy_score(y_test, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, preds, average="macro")

        model.save(MODEL_PATHS["deepnn"])
        joblib.dump(vectorizer, META_PATH)

    else:
        pipe = build_pipeline(model_type)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro")
        # Save classical pipeline inside a small dict for consistent loading
        joblib.dump({"pipeline": pipe}, MODEL_PATHS[model_type])

    cm = confusion_matrix(y_test, preds if model_type == "deepnn" else y_pred, labels=LABELS).tolist()
    return TrainResult(model_type, {"accuracy": acc, "precision": precision, "recall": recall, "f1_macro": f1}, cm, MODEL_PATHS[model_type])

# ---------------------------
# Auto-selection
# ---------------------------
def auto_select_model(data_dir: str) -> str:
    print("Training all models to select best performer...")
    results = {}
    for m in ["logreg", "rf", "deepnn"]:
        # Check for external stop flag before starting each candidate model
        try:
            if Path(STOP_FLAG).exists():
                print("Stop flag detected before starting next candidate model - aborting auto selection.")
                break
        except Exception:
            pass

        try:
            res = train_model(data_dir, m)
            results[m] = res.metrics
        except KeyboardInterrupt:
            print("Training was interrupted by stop flag. Stopping auto selection loop.")
            break
        except Exception as e:
            print(f"Error training model {m}: {e}")
            continue
    best = max(results, key=lambda k: 0.6 * results[k]["f1_macro"] + 0.4 * results[k]["accuracy"])
    print(f"Best model: {best}")
    joblib.dump({"best_model_type": best, "metrics": results}, SELECTION_PATH)
    return best


# ---------------------------
# Prediction utilities
# ---------------------------

def predict_with_confidence(model, text: str, model_type="logreg") -> Tuple[str, Dict[str, float]]:
    """
    Predict sensitivity class and confidence for text input.
    Supports classical ML and DeepNN.
    """
    import tensorflow as tf

    if model_type == "deepnn":
        meta = joblib.load(META_PATH)
        vectorizer = meta
        X_vec = vectorizer.transform([text]).toarray()
        deep_model = tf.keras.models.load_model(MODEL_PATHS["deepnn"])
        probs = deep_model.predict(X_vec, verbose=0)[0]
        conf = dict(zip(LABELS, np.round(probs, 3).tolist()))
        pred = LABELS[int(np.argmax(probs))]
        return pred, conf

    else:
        if isinstance(model, dict) and "pipeline" in model:
            model = model["pipeline"]
        if hasattr(model.named_steps["clf"], "predict_proba"):
            probs = model.predict_proba([text])[0]
            conf = dict(zip(model.named_steps["clf"].classes_, np.round(probs, 3)))
            pred = max(conf, key=conf.get)
        else:
            pred = model.predict([text])[0]
            conf = {pred: 1.0}
        return pred, conf


def hybrid_predict(text: str) -> Tuple[str, Dict[str, float]]:
    """
    Hybrid ensemble prediction combining classical (logreg) + deep model.
    """
    import tensorflow as tf

    # Load models
    ml_model = joblib.load(MODEL_PATHS["logreg"])
    deep_model = tf.keras.models.load_model(MODEL_PATHS["deepnn"])
    vectorizer = joblib.load(META_PATH)

    # Classical
    pred_ml, conf_ml = predict_with_confidence(ml_model, text, model_type="logreg")

    # Deep
    X_vec = vectorizer.transform([text]).toarray()
    probs = deep_model.predict(X_vec, verbose=0)[0]
    conf_dl = dict(zip(LABELS, probs.tolist()))
    pred_dl = LABELS[int(np.argmax(probs))]

    # Combine — average probabilities
    combined = {l: (conf_ml.get(l, 0) + conf_dl.get(l, 0)) / 2 for l in LABELS}
    pred_final = max(combined, key=combined.get)

    return pred_final, combined


def classify_text(text: str, use_model: bool = True) -> Dict[str, Any]:
    """High-level adapter used by the Streamlit UI to scrub + classify text.

    Returns a dict with keys:
    - scrubbed_text: the text with placeholders
    - placeholder_mapping: dict placeholder -> original
    - matches: summary of detected sensitive entity types (keys)
    - classification: predicted class (C1..C4)
    - confidence: confidence dict or heuristic score
    """
    ds = DynamicScrubber()
    scrubbed, placeholder_map = ds.scrub(text)

    # matches: just the placeholder types seen
    matches = {k: v for k, v in placeholder_map.items()}

    classification = None
    confidence = {}

    if use_model:
        # try to use model_selection to pick best model
        try:
            sel = joblib.load(SELECTION_PATH) if Path(SELECTION_PATH).exists() else None
            best = sel.get("best_model_type") if sel else None
        except Exception:
            best = None

        try:
            if best:
                if best == "deepnn":
                    classification, confidence = predict_with_confidence(None, text, model_type="deepnn")
                else:
                    model_obj = joblib.load(MODEL_PATHS.get(best))
                    classification, confidence = predict_with_confidence(model_obj, text, model_type=best)
            else:
                # fallback to logistic regression if available
                if Path(MODEL_PATHS["logreg"]).exists():
                    model_obj = joblib.load(MODEL_PATHS["logreg"])
                    classification, confidence = predict_with_confidence(model_obj, text, model_type="logreg")
        except Exception:
            classification = heuristic_label(text)
            confidence = {classification: 1.0}
    else:
        classification = heuristic_label(text)
        confidence = {classification: 1.0}

    return {
        "scrubbed_text": scrubbed,
        "placeholder_mapping": placeholder_map,
        "matches": matches,
        "classification": classification,
        "confidence": confidence,
    }

