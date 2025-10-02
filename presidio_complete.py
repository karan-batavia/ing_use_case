"""
Presidio Hybrid Anonymization + Classification Pipeline

- Combines regex-based, heuristic, semantic, and ML-based detection.
- Languages supported: English, French, Dutch (en, fr, nl).
- Uses Presidio Analyzer + Anonymizer for entity-level detection.
- Uses sensitivity_classifier.joblib (trained separately) for document-level classification (C1–C4).

Entity Categories:
- Regex entities: Emails, IBANs, SSNs, etc.
- Semantic entities: Religion, Ethnicity, Sexual Orientation.
- Heuristic entities: C1–C4 risk triggers (policy, agreement, credit score…).
- PERSON entities: Detected with spaCy (C3 risk level).
"""

from presidio_analyzer import (
    AnalyzerEngine,
    RecognizerRegistry,
    PatternRecognizer,
    EntityRecognizer,
    RecognizerResult,
)
from presidio_anonymizer import AnonymizerEngine
import spacy
import joblib
from pathlib import Path

# ----------------------
# Load SpaCy models
# ----------------------
# For multilingual: en, fr, nl
# Make sure you install these: python -m spacy download en_core_web_md fr_core_news_md nl_core_news_md
nlp_en = spacy.load("en_core_web_md")
nlp_fr = spacy.load("fr_core_news_md")
nlp_nl = spacy.load("nl_core_news_md")

# ----------------------
# Import regex patterns
# ----------------------
from regex_queries import ALL_PATTERNS   # your custom regex dictionary

# Import fallback ML regexes (from ml_setup.py if separated, or move to shared_patterns.py)
from patterns_shared import PATTERNS as ML_PATTERNS

# ----------------------
# Sensitive categories (semantic similarity)
# ----------------------
sensitive_categories = {
    "RELIGION": ["christian", "muslim", "jewish", "hindu", "buddhist", "atheist"],
    "ETHNICITY": ["asian", "african", "latino", "caucasian", "arab", "indigenous"],
    "SEXUAL_ORIENTATION": ["gay", "lesbian", "bisexual", "transgender", "queer", "lgbtq"],
}

category_embeddings = {
    category: [nlp_en(term)[0].vector for term in terms]
    for category, terms in sensitive_categories.items()
}

# ----------------------
# Custom Semantic Similarity Recognizer
# ----------------------
class SemanticSimilarityRecognizer(EntityRecognizer):
    def __init__(self, supported_entities=None, threshold=0.7):
        super().__init__(supported_entities or list(sensitive_categories.keys()), "en")
        self.threshold = threshold

    def analyze(self, text, entities, nlp_artifacts=None):
        results = []
        doc = nlp_en(text)
        for token in doc:
            if not token.is_alpha:
                continue
            for category, vectors in category_embeddings.items():
                for v in vectors:
                    similarity = token.vector.dot(v) / (
                        (token.vector_norm * (v**2) ** 0.5) + 1e-9
                    )
                    if similarity > self.threshold:
                        results.append(
                            RecognizerResult(
                                entity_type=category,
                                start=token.idx,
                                end=token.idx + len(token.text),
                                score=float(similarity),
                            )
                        )
        return results

# ----------------------
# Heuristic Recognizer (from ML heuristics)
# ----------------------
class HeuristicRecognizer(EntityRecognizer):
    def __init__(self):
        super().__init__(supported_entities=["C1", "C2", "C3", "C4"], supported_language="en")

    def analyze(self, text, entities, nlp_artifacts=None):
        results = []
        t = text.lower()
        if any(w in t for w in ["credit score", "income", "account balance", "masked pin", "biometric"]):
            results.append(RecognizerResult("C4", 0, len(text), 0.9))
        elif any(w in t for w in ["agreement", "supplier", "customer", "invoice", "payment order"]):
            results.append(RecognizerResult("C3", 0, len(text), 0.8))
        elif any(w in t for w in ["policy", "guideline", "governance", "standard", "sop"]):
            results.append(RecognizerResult("C2", 0, len(text), 0.7))
        elif any(w in t for w in ["press", "annual report", "newsroom", "investor"]):
            results.append(RecognizerResult("C1", 0, len(text), 0.6))
        return results

# ----------------------
# PERSON Recognizer (using spaCy NER)
# ----------------------
class SpacyPersonRecognizer(EntityRecognizer):
    def __init__(self, nlp_model, lang_code):
        super().__init__(supported_entities=["PERSON"], supported_language=lang_code)
        self.nlp = nlp_model

    def analyze(self, text, entities, nlp_artifacts=None):
        results = []
        doc = self.nlp(text)
        for ent in doc.ents:
            if ent.label_ == "PER" or ent.label_ == "PERSON":  # "PER" in FR/NL models
                results.append(
                    RecognizerResult(
                        entity_type="C3_PERSON",  # treat as C3-level sensitive
                        start=ent.start_char,
                        end=ent.end_char,
                        score=0.85,
                    )
                )
        return results

# ----------------------
# Setup Presidio Analyzer + Registry
# ----------------------
registry = RecognizerRegistry()

# Register regex recognizers from ALL_PATTERNS
for entity, pattern in ALL_PATTERNS.items():
    recognizer = PatternRecognizer(
        supported_entity=entity.upper(),
        patterns=[{"name": entity, "pattern": pattern, "score": 0.8}],
    )
    registry.add_recognizer(recognizer)

# Register regex recognizers from ML regexes
for entity, pattern in ML_PATTERNS.items():
    recognizer = PatternRecognizer(
        supported_entity=entity.upper(),
        patterns=[{"name": entity, "pattern": pattern, "score": 0.8}],
    )
    registry.add_recognizer(recognizer)

# Register semantic, heuristic, and person recognizers
registry.add_recognizer(SemanticSimilarityRecognizer())
registry.add_recognizer(HeuristicRecognizer())
registry.add_recognizer(SpacyPersonRecognizer(nlp_en, "en"))
registry.add_recognizer(SpacyPersonRecognizer(nlp_fr, "fr"))
registry.add_recognizer(SpacyPersonRecognizer(nlp_nl, "nl"))

# Initialize Presidio engines
analyzer = AnalyzerEngine(registry=registry, supported_languages=["en", "fr", "nl"])
anonymizer = AnonymizerEngine()

# ----------------------
# Load ML sensitivity classifier
# ----------------------
MODEL_PATH = Path(__file__).parent / "sensitivity_classifier.joblib"
clf_bundle = joblib.load(MODEL_PATH)
classifier = clf_bundle["pipeline"]
labels = clf_bundle["labels"]

# ----------------------
# Combined pipeline
# ----------------------
def analyze_and_anonymize_with_classification(text: str, lang="en"):
    """
    Full pipeline:
    1. Detect sensitive entities (regex + semantic + heuristics + PERSON).
    2. Anonymize detected entities.
    3. Run ML classifier to assign document-level sensitivity (C1–C4).
    """
    # Step 1: Entity analysis
    results = analyzer.analyze(text=text, language=lang)
    anonymized_text = anonymizer.anonymize(text=text, analyzer_results=results)

    # Step 2: Sensitivity classification
    sensitivity = classifier.predict([text])[0]

    return {
        "original": text,
        "anonymized": anonymized_text.text,
        "entities": [r.to_dict() for r in results],
        "sensitivity": sensitivity,
    }

# ----------------------
# Example usage
# ----------------------
if __name__ == "__main__":
    sample = "John is a Christian engineer. Contact him at john.doe@email.com. Rapport interne politique."
    output = analyze_and_anonymize_with_classification(sample, lang="en")

    print("=== Presidio + ML Pipeline ===")
    print("Original:   ", output["original"])
    print("Anonymized: ", output["anonymized"])
    print("Entities:   ", output["entities"])
    print("Sensitivity:", output["sensitivity"])
