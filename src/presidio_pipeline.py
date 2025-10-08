"""
Presidio-based Anonymization + Sensitivity Classification Pipeline

- Uses Presidio (Analyzer + Anonymizer) for entity detection and redaction.
- Adds regex-based and semantic recognizers for custom entities.
- After anonymization, applies your ML classifier (C1–C4) from sensitivity_classifier.joblib.
"""

from presidio_analyzer import AnalyzerEngine, RecognizerRegistry, PatternRecognizer
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
import joblib
from pathlib import Path

# ----------------------
# Import regex patterns
# ----------------------
from preprocessing.regex_queries import ALL_PATTERNS

# ----------------------
# Optional: custom recognizers and mappings
# ----------------------
try:
    from .recognizers import SemanticSimilarityRecognizer, HeuristicRecognizer, PERSON_RECOGNIZERS
    from .entity_mapping import ENTITY_SENSITIVITY
except ImportError:
    # Fallbacks if those modules aren't defined yet
    SemanticSimilarityRecognizer = HeuristicRecognizer = None
    PERSON_RECOGNIZERS = []
    ENTITY_SENSITIVITY = {}

# ----------------------
# Setup recognizer registry
# ----------------------
registry = RecognizerRegistry()

# Add regex recognizers from ALL_PATTERNS
for entity, pattern in ALL_PATTERNS.items():
    recognizer = PatternRecognizer(
        supported_entity=entity.upper(),
        patterns=[{"name": entity, "pattern": pattern, "score": 0.8}],
    )
    registry.add_recognizer(recognizer)

# Add optional semantic + heuristic + PERSON recognizers
if SemanticSimilarityRecognizer:
    registry.add_recognizer(SemanticSimilarityRecognizer())
if HeuristicRecognizer:
    registry.add_recognizer(HeuristicRecognizer())
for person_rec in PERSON_RECOGNIZERS:
    registry.add_recognizer(person_rec)

# Initialize Presidio analyzer and anonymizer
analyzer = AnalyzerEngine(registry=registry, supported_languages=["en"])
anonymizer = AnonymizerEngine()

# ----------------------
# Define anonymization config
# ----------------------
ANONYMIZER_CONFIG = {
    "DEFAULT": OperatorConfig("replace", {"new_value": "[REDACTED]"}),

    "PERSON": OperatorConfig("replace", {"new_value": "[PERSON]"}),
    "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "[EMAIL]"}),
    "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "[PHONE]"}),
    "IBAN_CODE": OperatorConfig("replace", {"new_value": "[IBAN]"}),
    "CREDIT_CARD": OperatorConfig("replace", {"new_value": "[CREDIT_CARD]"}),
    "CRYPTO": OperatorConfig("replace", {"new_value": "[CRYPTO]"}),
    "IP_ADDRESS": OperatorConfig("replace", {"new_value": "[IP]"}),
    "URL": OperatorConfig("replace", {"new_value": "[URL]"}),
    "LOCATION": OperatorConfig("replace", {"new_value": "[LOCATION]"}),
    "DATE_TIME": OperatorConfig("replace", {"new_value": "[DATE]"}),

    # Project-specific entities (optional)
    "SOCIAL_SECURITY": OperatorConfig("replace", {"new_value": "[SSN]"}),
    "PIN": OperatorConfig("replace", {"new_value": "[PIN]"}),
    "CVV": OperatorConfig("replace", {"new_value": "[CVV]"}),
    "TRANSACTION": OperatorConfig("replace", {"new_value": "[TRANSACTION]"}),
    "CUSTOMER_NUMBER": OperatorConfig("replace", {"new_value": "[CUSTOMER_NO]"}),
    "NATIONAL_ID": OperatorConfig("replace", {"new_value": "[NATIONAL_ID]"}),
    "ADDRESS": OperatorConfig("replace", {"new_value": "[ADDRESS]"}),
    "EMPLOYEE_ID": OperatorConfig("replace", {"new_value": "[EMP_ID]"}),
    "CONTRACT_NUMBER": OperatorConfig("replace", {"new_value": "[CONTRACT]"}),
}

# ----------------------
# Load ML sensitivity classifier
# ----------------------
MODEL_PATH = Path(__file__).parent / "sensitivity_classifier.joblib"
clf_bundle = joblib.load(MODEL_PATH)
classifier = clf_bundle["pipeline"]
labels = clf_bundle["labels"]

# ----------------------
# Helper: filter false positives (optional)
# ----------------------
EXCLUDE_TERMS = {"Full Year Results", "Annual Report", "Press Release"}

def filter_presidio_results(results):
    filtered = []
    for r in results:
        if getattr(r, "text", "").strip() in EXCLUDE_TERMS:
            continue
        filtered.append(r)
    return filtered

# ----------------------
# Combined pipeline
# ----------------------
def analyze_and_anonymize_with_classification(text: str, lang="en"):
    """
    1. Detect and anonymize sensitive entities using Presidio.
    2. Predict sensitivity level (C1–C4) using the ML model.
    """
    results = analyzer.analyze(text=text, language=lang)
    results = filter_presidio_results(results)

    anonymized = anonymizer.anonymize(
        text=text,
        analyzer_results=results,
        operators=ANONYMIZER_CONFIG
    )

    entities_with_levels = []
    for r in results:
        mapped_level = ENTITY_SENSITIVITY.get(r.entity_type, "UNKNOWN")
        e = r.to_dict()
        e["mapped_level"] = mapped_level
        entities_with_levels.append(e)

    sensitivity = classifier.predict([text])[0]

    return {
        "original": text,
        "anonymized": anonymized.text,
        "entities": entities_with_levels,
        "sensitivity": sensitivity,
    }

# ----------------------
# Example usage
# ----------------------
if __name__ == "__main__":
    sample = "Give the account number of customer John Smith with ID 901212-66616"
    result = analyze_and_anonymize_with_classification(sample)

    print("=== Presidio + ML Pipeline ===")
    print("Original:", result["original"])
    print("Anonymized:", result["anonymized"])
    print("Entities:", result["entities"])
    print("Sensitivity:", result["sensitivity"])