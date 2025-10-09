"""
Sensitivity Classification Service
Handles text classification and sensitive data redaction
"""

import os
import re
import logging
import json
import joblib
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from sklearn.base import BaseEstimator, TransformerMixin
from dataclasses import dataclass
from src.rule_featureizer import RuleFeatureizer 

logger = logging.getLogger(__name__)

# Patterns for sensitive data detection
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

PATTERNS = {k: re.compile(v, re.IGNORECASE) for k, v in FALLBACK_PATTERNS.items()}

# Placeholders for each pattern type
PLACEHOLDERS = {
    "EMAIL": "[EMAIL]",
    "PHONE_EU": "[PHONE]",
    "SSN_LIKE": "[SSN]",
    "IBAN": "[IBAN]",
    "ACCOUNT_NUM": "[ACCOUNT_NUMBER]",
    "AMOUNT": "[AMOUNT]",
    "DOB": "[DATE_OF_BIRTH]",
    "NATIONAL_ID": "[NATIONAL_ID]",
    "BIOMETRIC": "[BIOMETRIC_DATA]",
}


class RuleFeatureizer(BaseEstimator, TransformerMixin):
    """Rule-based feature extractor - must match ml_setup.py exactly"""

    def __init__(self, pattern_keys: Optional[List[str]] = None):
        self.pattern_keys = pattern_keys or list(PATTERNS.keys())

    def fit(self, X: List[str], y=None):
        return self

    def transform(self, X: List[str]) -> Any:
        feats = []
        for text in X:
            row = []
            for k in self.pattern_keys:
                p = PATTERNS[k]
                matches = p.findall(text)
                row.append(int(bool(matches)))  # presence
                row.append(len(matches))  # count

            def has_any(words):
                tl = text.lower()
                return int(any(w in tl for w in words))

            row.extend(
                [
                    has_any(
                        [
                            "credit score",
                            "income",
                            "account balance",
                            "masked pin",
                            "biometric",
                        ]
                    ),
                    has_any(
                        [
                            "agreement",
                            "supplier",
                            "customer",
                            "standing order",
                            "payment order",
                            "overdraft",
                        ]
                    ),
                    has_any(
                        [
                            "annual report",
                            "pillar 3",
                            "press",
                            "newsroom",
                            "full year results",
                            "investor",
                        ]
                    ),
                    has_any(
                        [
                            "policy",
                            "guideline",
                            "standard",
                            "sop",
                            "governance",
                            "raci",
                            "deprecated",
                            "retired",
                        ]
                    ),
                ]
            )
            feats.append(row)
        return np.array(feats, dtype=float)


@dataclass
class ClassificationResult:
    """Result of text classification"""

    prediction: str
    probabilities: Dict[str, float]
    confidence: float
    explanation: str


@dataclass
class RedactionResult:
    """Result of text redaction"""

    original_text: str
    redacted_text: str
    detections: List[Dict[str, str]]
    total_redacted: int
    detection_summary: Dict[str, int]


@dataclass
class ModelMetrics:
    """Model performance metrics"""

    accuracy: float
    precision: Dict[str, float]
    recall: Dict[str, float]
    f1_score: Dict[str, float]
    support: Dict[str, int]
    confusion_matrix: List[List[int]]



class SensitivityClassifierService:
    """Service for sensitivity classification and redaction"""

    def __init__(self, model_path: str = "sensitivity_classifier.joblib"):
        # Try to resolve the model path robustly
        root_dir = Path(__file__).resolve().parent  # src/
        possible_paths = [
            Path(model_path),                            # relative (fallback)
            root_dir / model_path,                       # src/sensitivity_classifier.joblib
            root_dir.parent / model_path,                # ing_use_case/sensitivity_classifier.joblib
            root_dir.parent / "logs" / model_path,       # ing_use_case/logs/sensitivity_classifier.joblib
        ]

        # Pick the first one that exists
        self.model_path = next((p for p in possible_paths if p.exists()), possible_paths[0])

        self.pipeline = None
        self.labels = None
        self._load_model()

        # Category explanations
        self.explanations = {
            "C1": "📄 Public information suitable for external sharing",
            "C2": "🏢 Internal use only - policies and procedures",
            "C3": "🔐 Confidential - business transactions",
            "C4": "⚠️ Highly Sensitive - contains PII or financial data",
        }

    def _load_model(self):
        """Load the trained model"""
        if not self.model_path.exists():
            logger.warning(f"[CLASSIFIER] Model file not found at: {self.model_path.resolve()}")
            return

        try:
            data = joblib.load(self.model_path)
            self.pipeline = data.get("pipeline")
            self.labels = data.get("labels")
            logger.info(f"[CLASSIFIER] Loaded model successfully from: {self.model_path}")
        except Exception as e:
            logger.error(f"[CLASSIFIER] Error loading model: {e}")
            self.pipeline = None
            self.labels = None
    def is_model_available(self) -> bool:
        """Check if model is loaded and available"""
        return self.pipeline is not None and self.labels is not None

    def classify_text(self, text: str) -> ClassificationResult:
        """Classify text sensitivity level"""
        if not self.is_model_available():
            raise ValueError("Model not available. Please train the model first.")

        try:
            # Get prediction and probabilities
            prediction = self.pipeline.predict([text])[0]
            probabilities_array = self.pipeline.predict_proba([text])[0]

            # Convert to dictionary
            probabilities = {
                label: float(prob)
                for label, prob in zip(self.labels, probabilities_array)
            }

            # Calculate confidence (max probability)
            confidence = float(max(probabilities_array))

            # Get explanation
            explanation = self.explanations.get(prediction, "")

            return ClassificationResult(
                prediction=prediction,
                probabilities=probabilities,
                confidence=confidence,
                explanation=explanation,
            )

        except Exception as e:
            logger.error(f"Error classifying text: {e}")
            raise

    def redact_sensitive_info(self, text: str) -> RedactionResult:
        """Redact sensitive information from text"""
        try:
            redacted_text = text
            detections = []

            # Track all matches with their positions (in reverse order to maintain indices)
            matches = []
            for pattern_name, pattern in PATTERNS.items():
                for match in pattern.finditer(text):
                    matches.append(
                        {
                            "start": match.start(),
                            "end": match.end(),
                            "text": match.group(),
                            "type": pattern_name,
                            "placeholder": PLACEHOLDERS.get(
                                pattern_name, f"[{pattern_name}]"
                            ),
                        }
                    )

            # Sort by position (reverse order to maintain string indices during replacement)
            matches.sort(key=lambda x: x["start"], reverse=True)

            # Replace matches with placeholders
            for match in matches:
                redacted_text = (
                    redacted_text[: match["start"]]
                    + match["placeholder"]
                    + redacted_text[match["end"] :]
                )
                detections.append(
                    {
                        "type": match["type"],
                        "original": match["text"],
                        "placeholder": match["placeholder"],
                    }
                )

            # Reverse detections to show in original order
            detections.reverse()

            # Create detection summary
            detection_summary = {}
            for detection in detections:
                det_type = detection["type"]
                detection_summary[det_type] = detection_summary.get(det_type, 0) + 1

            return RedactionResult(
                original_text=text,
                redacted_text=redacted_text,
                detections=detections,
                total_redacted=len(detections),
                detection_summary=detection_summary,
            )

        except Exception as e:
            logger.error(f"Error redacting text: {e}")
            raise

    def get_model_metrics(self) -> Optional[ModelMetrics]:
        """Get model performance metrics"""
        metrics_file = Path("model_metrics.json")

        if not metrics_file.exists():
            return None

        try:
            with open(metrics_file, "r") as f:
                metrics_data = json.load(f)

            # Parse classification report
            report_text = metrics_data.get("report", "")
            confusion_matrix = metrics_data.get("confusion_matrix", [])

            # Extract metrics from report text
            lines = report_text.strip().split("\n")
            precision = {}
            recall = {}
            f1_score = {}
            support = {}
            accuracy = 0.0

            for line in lines:
                parts = line.split()
                if len(parts) >= 5 and parts[0] in ["C1", "C2", "C3", "C4"]:
                    category = parts[0]
                    precision[category] = float(parts[1])
                    recall[category] = float(parts[2])
                    f1_score[category] = float(parts[3])
                    support[category] = int(parts[4])
                elif "accuracy" in line.lower() and len(parts) >= 2:
                    accuracy = float(parts[1])

            return ModelMetrics(
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1_score,
                support=support,
                confusion_matrix=confusion_matrix,
            )

        except Exception as e:
            logger.error(f"Error loading model metrics: {e}")
            return None

    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        return ["logreg", "rf"]

    def get_category_info(self) -> Dict[str, str]:
        """Get information about classification categories"""
        return {
            "C1": "Public (reports, press)",
            "C2": "Internal (policies, SOPs)",
            "C3": "Confidential (transactions)",
            "C4": "Highly Sensitive (PII, financial)",
        }


# Singleton instance
_classifier_service: Optional[SensitivityClassifierService] = None


def get_classifier_service() -> SensitivityClassifierService:
    """Get or create singleton instance of SensitivityClassifierService"""
    global _classifier_service

    if _classifier_service is None:
        _classifier_service = SensitivityClassifierService()

    return _classifier_service
