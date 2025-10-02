"""
Presidio Anonymization Pipeline

- Uses Presidio Analyzer + Anonymizer with your custom regexes and semantic recognizer.
- After anonymization, runs ML classifier (from sensitivity_classifier.joblib)
  to assign a sensitivity class (C1–C4) to the whole document.

How it fits:
- ML (ml_setup.py) = sensitivity classifier (C1–C4).
- Presidio (this file) = entity-level anonymization (PII, sensitive categories).
"""

from presidio_analyzer import AnalyzerEngine, RecognizerRegistry, PatternRecognizer, RecognizerResult, EntityRecognizer
from presidio_anonymizer import AnonymizerEngine
import spacy
import joblib
from pathlib import Path

# ----------------------
# Load SpaCy for embeddings
# ----------------------
nlp = spacy.load("en_core_web_md")

# ----------------------
# Import regex patterns (already defined in regex_queries.py)
# ----------------------
from regex_queries import ALL_PATTERNS

# ----------------------
# Sensitive categories (semantic similarity)
# ----------------------
sensitive_categories = {
    "RELIGION": ["christian", "muslim", "jewish", "hindu", "buddhist", "atheist"],
    "ETHNICITY": ["asian", "african", "latino", "caucasian", "arab", "indigenous"],
    "SEXUAL_ORIENTATION": ["gay", "lesbian", "bisexual", "transgender", "queer", "lgbtq"],
}

category_embeddings = {
    category: [nlp(term)[0].vector for term in terms]
    for category, terms in sensitive_categories.items()
}

# ----------------------
# Custom semantic similarity recognizer
# ----------------------
class SemanticSimilarityRecognizer(EntityRecognizer):
    def __init__(self, supported_entities=None, threshold=0.7):
        super().__init__(supported_entities or list(sensitive_categories.keys()), "en")
        self.threshold = threshold

    def analyze(self, text, entities, nlp_artifacts=None):
        results = []
        doc = nlp(text)

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
# Setup Presidio Analyzer + Registry
# ----------------------
registry = RecognizerRegistry()

# Register regex recognizers
for entity, pattern in ALL_PATTERNS.items():
    recognizer = PatternRecognizer(
        supported_entity=entity.upper(),
        patterns=[{"name": entity, "pattern": pattern, "score": 0.8}],
    )
    registry.add_recognizer(recognizer)

# Register semantic recognizer
registry.add_recognizer(SemanticSimilarityRecognizer())

# Initialize Presidio engines
analyzer = AnalyzerEngine(registry=registry, supported_languages=["en"])
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
def analyze_and_anonymize_with_classification(text: str):
    """
    1. Use Presidio to detect and anonymize entities.
    2. Use ML classifier to assign sensitivity label (C1–C4).
    """
    # Step 1: Presidio analysis
    results = analyzer.analyze(text=text, language="en")
    anonymized_text = anonymizer.anonymize(text=text, analyzer_results=results)

    # Step 2: Sensitivity classification (on original or anonymized text)
    # Choice: classify original text (more accurate)
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
    sample = "John is a Christian engineer. Contact him at john.doe@email.com"
    output = analyze_and_anonymize_with_classification(sample)

    print("=== Presidio + ML Pipeline ===")
    print("Original:   ", output["original"])
    print("Anonymized: ", output["anonymized"])
    print("Entities:   ", output["entities"])
    print("Sensitivity:", output["sensitivity"])
