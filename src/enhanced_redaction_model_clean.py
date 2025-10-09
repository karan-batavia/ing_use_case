#!/usr/bin/env python3
"""
Enhanced Redaction Model for Banking Sensitive Data
Combines NER, regex patterns, and classification for superior redaction
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np
import pandas as pd
from datetime import datetime

# ML libraries
try:
    import spacy

    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import joblib

logger = logging.getLogger(__name__)


@dataclass
class EntityDetection:
    """Represents a detected sensitive entity"""

    text: str
    label: str
    start: int
    end: int
    confidence: float
    detection_method: str  # "ner", "regex", "hybrid"


@dataclass
class RedactionResult:
    """Enhanced redaction result with detailed tracking"""

    original_text: str
    redacted_text: str
    detections: List[EntityDetection]
    sensitivity_level: str
    confidence: float
    total_redacted: int
    detection_summary: Dict[str, int]
    method_summary: Dict[str, int]


class EnhancedBankingNER:
    """Banking-specific Named Entity Recognition with custom patterns"""

    def __init__(self):
        self.nlp = None
        self.banking_patterns = self._define_banking_patterns()
        self.confidence_threshold = 0.8
        if SPACY_AVAILABLE:
            self._load_spacy_model()

    def _load_spacy_model(self):
        """Load spaCy model with fallback options"""
        if not SPACY_AVAILABLE:
            logger.warning("spaCy not available - NER features disabled")
            return

        models_to_try = ["en_core_web_lg", "en_core_web_md", "en_core_web_sm"]

        for model_name in models_to_try:
            try:
                self.nlp = spacy.load(model_name)
                logger.info(f"Loaded spaCy model: {model_name}")
                break
            except OSError:
                continue

        if self.nlp is None:
            try:
                self.nlp = spacy.blank("en")
                logger.warning("Using blank spaCy model - NER features limited")
            except Exception:
                logger.error("Cannot initialize spaCy model")
                return

        # Add custom banking entity ruler
        self._add_banking_patterns()

    def _define_banking_patterns(self) -> Dict[str, List[Dict]]:
        """Define banking-specific entity patterns"""
        return {
            "IBAN": [
                {
                    "pattern": [{"TEXT": {"REGEX": r"[A-Z]{2}\d{2}[A-Z0-9]{11,30}"}}],
                    "label": "IBAN",
                },
            ],
            "ACCOUNT_NUMBER": [
                {
                    "pattern": [{"LOWER": "account"}, {"IS_DIGIT": True}],
                    "label": "ACCOUNT_NUMBER",
                },
                {
                    "pattern": [{"LOWER": "acct"}, {"IS_DIGIT": True}],
                    "label": "ACCOUNT_NUMBER",
                },
            ],
            "AMOUNT": [
                {
                    "pattern": [
                        {"TEXT": {"REGEX": r"(USD|EUR|GBP|€|\$|£)"}},
                        {"IS_DIGIT": True},
                    ],
                    "label": "MONEY",
                },
            ],
            "PHONE_BANKING": [
                {
                    "pattern": [
                        {
                            "TEXT": {
                                "REGEX": r"\+\d{1,3}[\s-]?\d{2}[\s-]?\d{3}[\s-]?\d{3}[\s-]?\d{3}"
                            }
                        }
                    ],
                    "label": "PHONE",
                },
            ],
            "TRANSACTION_ID": [
                {
                    "pattern": [
                        {
                            "TEXT": {
                                "REGEX": r"[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}"
                            }
                        }
                    ],
                    "label": "TRANSACTION_ID",
                },
            ],
        }

    def _add_banking_patterns(self):
        """Add banking patterns to spaCy pipeline"""
        if self.nlp is None:
            return

        try:
            # Check if spaCy version supports entity ruler
            if hasattr(self.nlp, "pipe_names") and hasattr(self.nlp, "add_pipe"):
                if "entity_ruler" not in self.nlp.pipe_names:
                    entity_ruler = self.nlp.add_pipe("entity_ruler", last=True)
                else:
                    entity_ruler = self.nlp.get_pipe("entity_ruler")

                # Add all patterns
                patterns = []
                for entity_type, pattern_list in self.banking_patterns.items():
                    patterns.extend(pattern_list)

                if hasattr(entity_ruler, "add_patterns"):
                    entity_ruler.add_patterns(patterns)
        except Exception as e:
            logger.warning(f"Could not add banking patterns: {e}")

    def extract_entities(self, text: str) -> List[EntityDetection]:
        """Extract entities using NER"""
        if self.nlp is None:
            return []

        try:
            doc = self.nlp(text)
            detections = []

            for ent in doc.ents:
                banking_label = self._map_spacy_label(ent.label_)
                if banking_label:
                    detection = EntityDetection(
                        text=ent.text,
                        label=banking_label,
                        start=ent.start_char,
                        end=ent.end_char,
                        confidence=self._calculate_confidence(ent),
                        detection_method="ner",
                    )
                    detections.append(detection)

            return detections
        except Exception as e:
            logger.error(f"Error in NER extraction: {e}")
            return []

    def _map_spacy_label(self, spacy_label: str) -> Optional[str]:
        """Map spaCy entity labels to banking-specific labels"""
        mapping = {
            "PERSON": "PERSON_NAME",
            "ORG": "ORGANIZATION",
            "MONEY": "AMOUNT",
            "DATE": "DATE",
            "TIME": "TIME",
            "PHONE_NUMBER": "PHONE",
            "EMAIL": "EMAIL",
            "IBAN": "IBAN",
            "ACCOUNT_NUMBER": "ACCOUNT_NUMBER",
            "TRANSACTION_ID": "TRANSACTION_ID",
        }
        return mapping.get(spacy_label)

    def _calculate_confidence(self, ent) -> float:
        """Calculate confidence score for entity detection"""
        if hasattr(ent, "label_") and ent.label_ in [
            "IBAN",
            "ACCOUNT_NUMBER",
            "TRANSACTION_ID",
        ]:
            return 0.95

        confidence_map = {
            "PERSON": 0.85,
            "ORG": 0.80,
            "MONEY": 0.90,
            "DATE": 0.75,
            "PHONE_NUMBER": 0.85,
            "EMAIL": 0.95,
        }

        return confidence_map.get(ent.label_, 0.70)


class RegexPatternMatcher:
    """Enhanced regex pattern matching for banking entities"""

    def __init__(self):
        self.patterns = self._define_enhanced_patterns()

    def _define_enhanced_patterns(self) -> Dict[str, Dict]:
        """Define comprehensive regex patterns for banking data based on detailed C1-C4 schema"""
        return {
            # =============================================================================
            # C4 SECRET - Customer sensitive data, Authentication, Credit card fraud data
            # =============================================================================
            # Personal Sensitive Data (C4)
            "ETHNIC_ORIGIN": {
                "pattern": re.compile(
                    r"\b(ethnicity|ethnic\s+origin|race|racial|black|white|caucasian|asian|african|hispanic|latino|indigenous|aboriginal|native|mixed\s+race|multiracial|biracial|italian|german|french|spanish|british|american|dutch|belgian|portuguese|greek|polish|romanian|hungarian|czech|slovak|bulgarian|croatian|serbian|slovenian|austrian|swiss|swedish|danish|norwegian|finnish|irish|scottish|welsh|russian|ukrainian|turkish|chinese|japanese|korean|indian|pakistani|bangladeshi|brazilian|argentinian|mexican|canadian|australian|south\s+african|arab|persian|jewish|roma|gypsy|inuit|maori)\b",
                    re.IGNORECASE,
                ),
                "confidence": 0.85,
                "placeholder": "[ETHNIC_ORIGIN]",
            },
            "RELIGION": {
                "pattern": re.compile(
                    r"\b(religion|religious|christian|muslim|jewish|hindu|buddhist|atheist|catholic|protestant|orthodox|faith|belief|church|mosque|synagogue|temple)\b",
                    re.IGNORECASE,
                ),
                "confidence": 0.85,
                "placeholder": "[RELIGIOUS_BELIEF]",
            },
            "POLITICAL_OPINION": {
                "pattern": re.compile(
                    r"\b(political|politics|party|democrat|republican|conservative|liberal|socialist|communist|votes|voting|election|campaign)\b",
                    re.IGNORECASE,
                ),
                "confidence": 0.80,
                "placeholder": "[POLITICAL_OPINION]",
            },
            "HEALTH_DATA": {
                "pattern": re.compile(
                    r"\b(health|medical|disability|disease|illness|medication|treatment|diagnosis|hospital|doctor|patient|surgery|therapy)\b",
                    re.IGNORECASE,
                ),
                "confidence": 0.90,
                "placeholder": "[HEALTH_DATA]",
            },
            "SEXUAL_ORIENTATION": {
                "pattern": re.compile(
                    r"\b(sexual\s+orientation|sexuality|heterosexual|homosexual|bisexual|transgender|lgbtq|gay|lesbian)\b",
                    re.IGNORECASE,
                ),
                "confidence": 0.90,
                "placeholder": "[SEXUAL_ORIENTATION]",
            },
            "TRADE_UNION": {
                "pattern": re.compile(
                    r"\b(trade\s+union|union\s+member|labor\s+union|workers\s+union|union\s+card|collective\s+bargaining)\b",
                    re.IGNORECASE,
                ),
                "confidence": 0.85,
                "placeholder": "[TRADE_UNION]",
            },
            "CRIMINAL_RECORD": {
                "pattern": re.compile(
                    r"\b(criminal|conviction|offence|arrest|jail|prison|court|guilty|crime|fraud|theft|felony|misdemeanor)\b",
                    re.IGNORECASE,
                ),
                "confidence": 0.90,
                "placeholder": "[CRIMINAL_DATA]",
            },
            "GENETIC_DATA": {
                "pattern": re.compile(
                    r"\b(genetic|dna|chromosome|gene|hereditary|genome|mutation|genetic\s+test)\b",
                    re.IGNORECASE,
                ),
                "confidence": 0.95,
                "placeholder": "[GENETIC_DATA]",
            },
            "BIOMETRIC": {
                "pattern": re.compile(
                    r"\b(biometric|fingerprint|face\s*id|facial\s+recognition|iris\s+scan|voice\s+print|palm\s+print|retina|biometry)\b",
                    re.IGNORECASE,
                ),
                "confidence": 0.95,
                "placeholder": "[BIOMETRIC_DATA]",
            },
            # Authentication Data (C4)
            "PIN": {
                "pattern": re.compile(r"\b(pin|PIN)[\s:]*\d{4,8}\b"),
                "confidence": 0.98,
                "placeholder": "[PIN]",
            },
            "PASSWORD": {
                "pattern": re.compile(
                    r"\b(password|passwd|pwd)[\s:]*[^\s]{4,}\b",
                    re.IGNORECASE,
                ),
                "confidence": 0.95,
                "placeholder": "[PASSWORD]",
            },
            # Credit Card and Fraud Data (C4)
            "CREDIT_CARD": {
                "pattern": re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"),
                "confidence": 0.95,
                "placeholder": "[CREDIT_CARD]",
            },
            "FRAUD_FLAG": {
                "pattern": re.compile(
                    r"\b(fraud|suspicious|risk\s+rating|warning|sanction|interdiction|blacklist|watchlist)\b",
                    re.IGNORECASE,
                ),
                "confidence": 0.85,
                "placeholder": "[FRAUD_DATA]",
            },
            "WHISTLEBLOWER": {
                "pattern": re.compile(
                    r"\b(whistleblower|anonymous\s+report|internal\s+complaint|misconduct\s+report)\b",
                    re.IGNORECASE,
                ),
                "confidence": 0.90,
                "placeholder": "[WHISTLEBLOWER]",
            },
            # =============================================================================
            # C3 CONFIDENTIAL - Customer/Employee data, Financial data, Agreements
            # =============================================================================
            # Personal Identity Data (C3)
            "FIRST_NAME": {
                "pattern": re.compile(
                    r"\b(first\s+name|given\s+name|forename)[\s:]*([A-Z][a-z]{1,20})\b",
                    re.IGNORECASE,
                ),
                "confidence": 0.75,
                "placeholder": "[FIRST_NAME]",
            },
            "LAST_NAME": {
                "pattern": re.compile(
                    r"\b(last\s+name|surname|family\s+name)[\s:]*([A-Z][a-z]{1,30})\b",
                    re.IGNORECASE,
                ),
                "confidence": 0.75,
                "placeholder": "[LAST_NAME]",
            },
            "CNP": {
                "pattern": re.compile(r"\b(CNP|cnp)[\s:]*\d{13}\b"),
                "confidence": 0.95,
                "placeholder": "[CNP]",
            },
            "CIF": {
                "pattern": re.compile(r"\b(CIF|cif)[\s:]*[A-Z0-9]{8,15}\b"),
                "confidence": 0.95,
                "placeholder": "[CIF]",
            },
            "ID_SERIES": {
                "pattern": re.compile(
                    r"\b(ID|id)\s+series[\s:]*[A-Z]{2}\d{6,8}\b", re.IGNORECASE
                ),
                "confidence": 0.90,
                "placeholder": "[ID_SERIES]",
            },
            "POSTAL_ADDRESS": {
                "pattern": re.compile(
                    r"\b\d+\s+[A-Za-z\s]+(?:street|st|avenue|ave|road|rd|lane|ln|boulevard|blvd|drive|dr)\b",
                    re.IGNORECASE,
                ),
                "confidence": 0.80,
                "placeholder": "[ADDRESS]",
            },
            "PHONE_EU": {
                "pattern": re.compile(
                    r"(?:\+\d{1,3}\s?)?(?:\d[\s-]?){9,}", re.IGNORECASE
                ),
                "confidence": 0.85,
                "placeholder": "[PHONE]",
            },
            "EMAIL": {
                "pattern": re.compile(
                    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", re.IGNORECASE
                ),
                "confidence": 0.95,
                "placeholder": "[EMAIL]",
            },
            "DATE_OF_BIRTH": {
                "pattern": re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{2}/\d{2}/\d{4}\b"),
                "confidence": 0.80,
                "placeholder": "[DATE_OF_BIRTH]",
            },
            "AGE": {
                "pattern": re.compile(
                    r"\b(age|years\s+old)[\s:]*\d{1,3}\b", re.IGNORECASE
                ),
                "confidence": 0.70,
                "placeholder": "[AGE]",
            },
            "MARITAL_STATUS": {
                "pattern": re.compile(
                    r"\b(marital\s+status|married|single|divorced|widowed|separated)\b",
                    re.IGNORECASE,
                ),
                "confidence": 0.80,
                "placeholder": "[MARITAL_STATUS]",
            },
            "GENDER": {
                "pattern": re.compile(
                    r"\b(gender|male|female|m/f|sex)\b", re.IGNORECASE
                ),
                "confidence": 0.75,
                "placeholder": "[GENDER]",
            },
            "CITIZENSHIP": {
                "pattern": re.compile(
                    r"\b(citizenship|nationality|citizen\s+of|national\s+of|born\s+in|from\s+(italy|germany|france|spain|uk|usa|netherlands|belgium|portugal|greece|poland|romania|hungary|czech\s+republic|slovakia|bulgaria|croatia|serbia|slovenia|austria|switzerland|sweden|denmark|norway|finland|ireland|scotland|wales|russia|ukraine|turkey|china|japan|korea|india|pakistan|bangladesh|brazil|argentina|mexico|canada|australia|south\s+africa))\b",
                    re.IGNORECASE,
                ),
                "confidence": 0.80,
                "placeholder": "[CITIZENSHIP]",
            },
            # Financial Data (C3)
            "IBAN": {
                "pattern": re.compile(
                    r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b", re.IGNORECASE
                ),
                "confidence": 0.98,
                "placeholder": "[IBAN]",
            },
            "ACCOUNT_NUMBER": {
                "pattern": re.compile(
                    r"\b(?:account|acct)[\s:]*\d{4,}\b", re.IGNORECASE
                ),
                "confidence": 0.90,
                "placeholder": "[ACCOUNT_NUMBER]",
            },
            "AMOUNT": {
                "pattern": re.compile(
                    r"(?:USD|EUR|GBP|€|\$|£)\s?\d{1,3}(?:[,\s]\d{3})*(?:\.\d{2})?",
                    re.IGNORECASE,
                ),
                "confidence": 0.92,
                "placeholder": "[AMOUNT]",
            },
            "TRANSACTION_DATA": {
                "pattern": re.compile(
                    r"\b(transaction|payment|transfer|deposit|withdrawal)[\s:]*[A-Z0-9-]{6,}\b",
                    re.IGNORECASE,
                ),
                "confidence": 0.85,
                "placeholder": "[TRANSACTION_DATA]",
            },
            "CREDIT_HISTORY": {
                "pattern": re.compile(
                    r"\b(credit\s+score|credit\s+rating|fico|credit\s+history|creditworthiness)\b",
                    re.IGNORECASE,
                ),
                "confidence": 0.85,
                "placeholder": "[CREDIT_HISTORY]",
            },
            # Employment Data (C3)
            "EMPLOYMENT": {
                "pattern": re.compile(
                    r"\b(employer|employment|job\s+title|position|salary|income|occupation)\b",
                    re.IGNORECASE,
                ),
                "confidence": 0.75,
                "placeholder": "[EMPLOYMENT_DATA]",
            },
            "EDUCATION": {
                "pattern": re.compile(
                    r"\b(education|degree|university|college|school|bachelor|master|phd|diploma)\b",
                    re.IGNORECASE,
                ),
                "confidence": 0.70,
                "placeholder": "[EDUCATION]",
            },
            # Technical/Security Data (C3)
            "IP_ADDRESS": {
                "pattern": re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),
                "confidence": 0.95,
                "placeholder": "[IP_ADDRESS]",
            },
            "LOCATION_DATA": {
                "pattern": re.compile(
                    r"\b(latitude|longitude|GPS|coordinates|location)[\s:]*[-+]?\d+\.?\d*\b",
                    re.IGNORECASE,
                ),
                "confidence": 0.85,
                "placeholder": "[LOCATION_DATA]",
            },
            "SOURCE_CODE": {
                "pattern": re.compile(
                    r"\b(source\s+code|configuration|firewall\s+rule|access\s+list|system\s+log)\b",
                    re.IGNORECASE,
                ),
                "confidence": 0.80,
                "placeholder": "[SYSTEM_CONFIG]",
            },
            # =============================================================================
            # C2 RESTRICTED - Employee contact data, Product specs, Policies
            # =============================================================================
            "EMPLOYEE_CONTACT": {
                "pattern": re.compile(
                    r"\b(employee|staff|colleague)[\s:]*[A-Za-z\s]+(?:phone|email|contact)\b",
                    re.IGNORECASE,
                ),
                "confidence": 0.70,
                "placeholder": "[EMPLOYEE_CONTACT]",
            },
            "PRODUCT_SPEC": {
                "pattern": re.compile(
                    r"\b(product\s+specification|system\s+development|calculation|legal\s+option)\b",
                    re.IGNORECASE,
                ),
                "confidence": 0.75,
                "placeholder": "[PRODUCT_SPEC]",
            },
            "POLICY": {
                "pattern": re.compile(
                    r"\b(policy|guideline|process|procedure|SOP|standard\s+operating\s+procedure)\b",
                    re.IGNORECASE,
                ),
                "confidence": 0.70,
                "placeholder": "[POLICY]",
            },
            "INFRASTRUCTURE": {
                "pattern": re.compile(
                    r"\b(hypervisor|ESX|storage|network\s+interface|VM|virtual\s+machine|CMDB)\b",
                    re.IGNORECASE,
                ),
                "confidence": 0.80,
                "placeholder": "[INFRASTRUCTURE]",
            },
            # =============================================================================
            # Legacy patterns (maintaining compatibility)
            # =============================================================================
            "SSN_LIKE": {
                "pattern": re.compile(r"\b\d{6}[- ]?\d{2,4}[\.]?\d{0,2}\b"),
                "confidence": 0.88,
                "placeholder": "[SSN]",
            },
            "NATIONAL_ID": {
                "pattern": re.compile(r"\bID[:\s-]?[A-Z0-9]{6,}\b", re.IGNORECASE),
                "confidence": 0.85,
                "placeholder": "[NATIONAL_ID]",
            },
            "TRANSACTION_ID": {
                "pattern": re.compile(
                    r"\b[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}\b",
                    re.IGNORECASE,
                ),
                "confidence": 0.98,
                "placeholder": "[TRANSACTION_ID]",
            },
            "CORP_KEY": {
                "pattern": re.compile(r"\b[A-Z]{2}\d{2}[A-Z]{2}\b"),
                "confidence": 0.75,
                "placeholder": "[CORP_KEY]",
            },
        }

    def extract_entities(self, text: str) -> List[EntityDetection]:
        """Extract entities using regex patterns"""
        detections = []

        for label, config in self.patterns.items():
            pattern = config["pattern"]
            confidence = config["confidence"]

            for match in pattern.finditer(text):
                detection = EntityDetection(
                    text=match.group(),
                    label=label,
                    start=match.start(),
                    end=match.end(),
                    confidence=confidence,
                    detection_method="regex",
                )
                detections.append(detection)

        return detections


class SensitivityClassifier:
    """Classification model for sensitivity levels"""

    def __init__(self):
        self.pipeline = None
        self.labels = ["C1", "C2", "C3", "C4"]
        self.model_path = Path("sensitivity_model.joblib")

    def train(self, training_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Train the sensitivity classification model"""
        texts = [item["original_text"] for item in training_data]
        labels = [item["sensitivity_level"] for item in training_data]

        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )

        vectorizer = TfidfVectorizer(
            max_features=10000, ngram_range=(1, 3), stop_words="english", lowercase=True
        )

        model = RandomForestClassifier(
            n_estimators=100, random_state=42, class_weight="balanced"
        )

        pipeline = Pipeline([("vectorizer", vectorizer), ("classifier", model)])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        classification_rep = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred).tolist()

        model_data = {"pipeline": pipeline, "labels": self.labels}

        joblib.dump(model_data, self.model_path)

        # Handle the accuracy access properly
        accuracy = float(classification_rep.get("accuracy", 0.0))

        metrics = {
            "accuracy": accuracy,
            "classification_report": classification_rep,
            "confusion_matrix": conf_matrix,
            "train_size": len(X_train),
            "test_size": len(X_test),
        }

        self.pipeline = pipeline
        return metrics

    def load_model(self) -> bool:
        """Load trained model"""
        if not self.model_path.exists():
            return False

        try:
            model_data = joblib.load(self.model_path)
            self.pipeline = model_data["pipeline"]
            self.labels = model_data["labels"]
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

    def predict(self, text: str) -> Tuple[str, float]:
        """Predict sensitivity level"""
        if self.pipeline is None:
            return "C2", 0.5

        prediction = str(self.pipeline.predict([text])[0])
        probabilities = self.pipeline.predict_proba([text])[0]
        confidence = float(max(probabilities))

        return prediction, confidence


class EnhancedRedactionModel:
    """Main redaction model combining NER, regex, and classification"""

    def __init__(self):
        self.ner_model = EnhancedBankingNER()
        self.regex_matcher = RegexPatternMatcher()
        self.sensitivity_classifier = SensitivityClassifier()

        # Try to load existing model
        self.sensitivity_classifier.load_model()

        # Access level based redaction rules
        # Key principle: if user has Cx access, they can see C1 through Cx data
        # All data with higher classification gets redacted
        self.access_level_rules = {
            "C1": {  # C1 access: Public access only - redacts most sensitive data (C2, C3, C4)
                "allowed_levels": ["C1"],
                "redact_patterns": [
                    # C4 Secret data
                    "ETHNIC_ORIGIN",
                    "RELIGION",
                    "POLITICAL_OPINION",
                    "HEALTH_DATA",
                    "SEXUAL_ORIENTATION",
                    "TRADE_UNION",
                    "CRIMINAL_RECORD",
                    "GENETIC_DATA",
                    "BIOMETRIC",
                    "PIN",
                    "PASSWORD",
                    "CREDIT_CARD",
                    "FRAUD_FLAG",
                    "WHISTLEBLOWER",
                    # C3 Confidential data
                    "FIRST_NAME",
                    "LAST_NAME",
                    "CNP",
                    "CIF",
                    "ID_SERIES",
                    "POSTAL_ADDRESS",
                    "PHONE_EU",
                    "EMAIL",
                    "DATE_OF_BIRTH",
                    "AGE",
                    "MARITAL_STATUS",
                    "GENDER",
                    "CITIZENSHIP",
                    "IBAN",
                    "ACCOUNT_NUMBER",
                    "AMOUNT",
                    "TRANSACTION_DATA",
                    "CREDIT_HISTORY",
                    "EMPLOYMENT",
                    "EDUCATION",
                    "IP_ADDRESS",
                    "LOCATION_DATA",
                    "SOURCE_CODE",
                    "SSN_LIKE",
                    "NATIONAL_ID",
                    "TRANSACTION_ID",
                    # C2 Restricted data
                    "EMPLOYEE_CONTACT",
                    "PRODUCT_SPEC",
                    "POLICY",
                    "INFRASTRUCTURE",
                ],
            },
            "C2": {  # C2 access: Internal access - redacts confidential and secret data (C3, C4)
                "allowed_levels": ["C1", "C2"],
                "redact_patterns": [
                    # C4 Secret data
                    "ETHNIC_ORIGIN",
                    "RELIGION",
                    "POLITICAL_OPINION",
                    "HEALTH_DATA",
                    "SEXUAL_ORIENTATION",
                    "TRADE_UNION",
                    "CRIMINAL_RECORD",
                    "GENETIC_DATA",
                    "BIOMETRIC",
                    "PIN",
                    "PASSWORD",
                    "CREDIT_CARD",
                    "FRAUD_FLAG",
                    "WHISTLEBLOWER",
                    # C3 Confidential data
                    "FIRST_NAME",
                    "LAST_NAME",
                    "CNP",
                    "CIF",
                    "ID_SERIES",
                    "POSTAL_ADDRESS",
                    "PHONE_EU",
                    "EMAIL",
                    "DATE_OF_BIRTH",
                    "AGE",
                    "MARITAL_STATUS",
                    "GENDER",
                    "CITIZENSHIP",
                    "IBAN",
                    "ACCOUNT_NUMBER",
                    "AMOUNT",
                    "TRANSACTION_DATA",
                    "CREDIT_HISTORY",
                    "EMPLOYMENT",
                    "EDUCATION",
                    "IP_ADDRESS",
                    "LOCATION_DATA",
                    "SOURCE_CODE",
                    "SSN_LIKE",
                    "NATIONAL_ID",
                    "TRANSACTION_ID",
                ],
            },
            "C3": {  # C3 access: Confidential access - redacts only secret data (C4)
                "allowed_levels": ["C1", "C2", "C3"],
                "redact_patterns": [
                    # Only C4 Secret data gets redacted
                    "ETHNIC_ORIGIN",
                    "RELIGION",
                    "POLITICAL_OPINION",
                    "HEALTH_DATA",
                    "SEXUAL_ORIENTATION",
                    "TRADE_UNION",
                    "CRIMINAL_RECORD",
                    "GENETIC_DATA",
                    "BIOMETRIC",
                    "PIN",
                    "PASSWORD",
                    "CREDIT_CARD",
                    "FRAUD_FLAG",
                    "WHISTLEBLOWER",
                ],
            },
            "C4": {  # C4 access: Secret access - sees most data with minimal redaction
                "allowed_levels": ["C1", "C2", "C3", "C4"],
                "redact_patterns": [
                    # Only extremely sensitive items might be redacted
                    "PIN",
                    "PASSWORD",  # Keep authentication data redacted even for C4 users
                ],
            },
        }

    def train(self, training_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Train the complete redaction model"""
        logger.info("Training sensitivity classifier...")
        metrics = self.sensitivity_classifier.train(training_data)
        logger.info(f"Training completed with accuracy: {metrics['accuracy']:.3f}")
        return metrics

    def redact_text(
        self,
        text: str,
        user_access_level: Optional[str] = None,
        data_sensitivity: Optional[str] = None,
    ) -> RedactionResult:
        """Perform comprehensive text redaction based on user access level"""

        # Step 1: Classify data sensitivity if not provided
        if data_sensitivity is None:
            data_sensitivity, sensitivity_confidence = (
                self.sensitivity_classifier.predict(text)
            )
        else:
            sensitivity_confidence = 1.0

        # Step 2: Extract entities using both methods
        ner_detections = self.ner_model.extract_entities(text)
        regex_detections = self.regex_matcher.extract_entities(text)

        # Step 3: Merge and deduplicate detections
        all_detections = self._merge_detections(ner_detections + regex_detections)

        # Step 4: Filter based on user access level
        # If no user access level specified, assume C4 (full access)
        effective_access_level = user_access_level or "C4"
        filtered_detections = self._filter_by_access_level(
            all_detections, effective_access_level, data_sensitivity
        )

        # Step 5: Apply redaction
        redacted_text = self._apply_redaction(text, filtered_detections)

        # Step 6: Generate summary statistics
        detection_summary = self._generate_detection_summary(filtered_detections)
        method_summary = self._generate_method_summary(filtered_detections)

        return RedactionResult(
            original_text=text,
            redacted_text=redacted_text,
            detections=filtered_detections,
            sensitivity_level=data_sensitivity,
            confidence=sensitivity_confidence,
            total_redacted=len(filtered_detections),
            detection_summary=detection_summary,
            method_summary=method_summary,
        )

    def _merge_detections(
        self, detections: List[EntityDetection]
    ) -> List[EntityDetection]:
        """Merge overlapping detections, preferring higher confidence"""
        if not detections:
            return []

        detections.sort(key=lambda x: x.start)
        merged = []
        current = detections[0]

        for next_detection in detections[1:]:
            if next_detection.start <= current.end:
                if next_detection.confidence > current.confidence:
                    current = next_detection
                elif (
                    next_detection.confidence == current.confidence
                    and next_detection.detection_method == "ner"
                    and current.detection_method == "regex"
                ):
                    current = next_detection
            else:
                merged.append(current)
                current = next_detection

        merged.append(current)
        return merged

    def _filter_by_access_level(
        self,
        detections: List[EntityDetection],
        user_access_level: str,
        data_sensitivity: str,
    ) -> List[EntityDetection]:
        """Filter detections based on user access level and data sensitivity

        Logic:
        - If user has C1 access: can only see C1 data, all else gets redacted
        - If user has C2 access: can see C1+C2 data, C3+C4 gets redacted
        - If user has C3 access: can see C1+C2+C3 data, only C4 gets redacted
        - If user has C4 access: can see all data, no redaction needed
        """

        if user_access_level not in self.access_level_rules:
            # Default to most restrictive if unknown access level
            user_access_level = "C1"

        access_config = self.access_level_rules[user_access_level]
        allowed_levels = access_config["allowed_levels"]

        # If the data sensitivity is within user's access level, apply pattern-based redaction
        if data_sensitivity in allowed_levels:
            # Apply specific pattern redaction rules for this access level
            redact_patterns = access_config["redact_patterns"]
            return [d for d in detections if d.label in redact_patterns]
        else:
            # Data sensitivity exceeds user access - redact everything
            return detections

    def _apply_redaction(self, text: str, detections: List[EntityDetection]) -> str:
        """Apply redaction to text"""
        if not detections:
            return text

        detections.sort(key=lambda x: x.start, reverse=True)
        redacted_text = text

        for detection in detections:
            placeholder = self._get_placeholder(detection.label)
            redacted_text = (
                redacted_text[: detection.start]
                + placeholder
                + redacted_text[detection.end :]
            )

        return redacted_text

    def _get_placeholder(self, label: str) -> str:
        """Get appropriate placeholder for entity type"""
        placeholders = {
            "PERSON_NAME": "[PERSON]",
            "ORGANIZATION": "[ORGANIZATION]",
            "EMAIL": "[EMAIL]",
            "PHONE": "[PHONE]",
            "PHONE_EU": "[PHONE]",
            "IBAN": "[IBAN]",
            "ACCOUNT_NUMBER": "[ACCOUNT_NUMBER]",
            "AMOUNT": "[AMOUNT]",
            "SSN_LIKE": "[SSN]",
            "DATE_OF_BIRTH": "[DATE_OF_BIRTH]",
            "NATIONAL_ID": "[NATIONAL_ID]",
            "BIOMETRIC": "[BIOMETRIC_DATA]",
            "TRANSACTION_ID": "[TRANSACTION_ID]",
            "CORP_KEY": "[CORP_KEY]",
            "DATE": "[DATE]",
            "TIME": "[TIME]",
        }
        return placeholders.get(label, f"[{label}]")

    def _generate_detection_summary(
        self, detections: List[EntityDetection]
    ) -> Dict[str, int]:
        """Generate summary of detections by type"""
        summary = {}
        for detection in detections:
            summary[detection.label] = summary.get(detection.label, 0) + 1
        return summary

    def _generate_method_summary(
        self, detections: List[EntityDetection]
    ) -> Dict[str, int]:
        """Generate summary of detections by method"""
        summary = {}
        for detection in detections:
            summary[detection.detection_method] = (
                summary.get(detection.detection_method, 0) + 1
            )
        return summary


def create_enhanced_redaction_model() -> EnhancedRedactionModel:
    """Factory function to create and initialize the enhanced redaction model"""
    return EnhancedRedactionModel()


if __name__ == "__main__":
    # Example usage and testing
    model = create_enhanced_redaction_model()

    test_text = """
    Dear John Smith,
    
    Your account NL91ABNA0417164300 has been credited with EUR 5,234.56.
    Transaction ID: 550e8400-e29b-41d4-a716-446655440000
    
    For questions, contact us at support@ing.com or +31 20 123 4567.
    
    Best regards,
    ING Bank
    """

    # Test with different access levels
    print("=" * 60)
    print("ENHANCED REDACTION MODEL - ACCESS LEVEL TESTING")
    print("=" * 60)

    print("\nOriginal text:")
    print(test_text)

    # Test different user access levels
    access_levels = ["C1", "C2", "C3", "C4"]

    for access_level in access_levels:
        result = model.redact_text(test_text, user_access_level=access_level)

        print(f"\n{'='*50}")
        print(f"USER ACCESS LEVEL: {access_level}")
        print(f"DATA SENSITIVITY: {result.sensitivity_level}")
        print(f"CONFIDENCE: {result.confidence:.2f}")
        print(f"{'='*50}")
        print(f"Redacted text:")
        print(result.redacted_text)
        print(f"Total redacted: {result.total_redacted}")
        print(f"Detection summary: {result.detection_summary}")
        print(f"Method summary: {result.method_summary}")
