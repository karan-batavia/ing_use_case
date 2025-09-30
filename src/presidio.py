"""
Custom Anonymizer Script using Presidio + SpaCy
------------------------------------------------
This script combines:
- Presidio for regex-based detection of sensitive entities (IBAN, credit card, etc.).
- A custom semantic similarity recognizer using SpaCy to detect sensitive categories
  such as religion, ethnicity, and sexual orientation.
- Automatic integration of your regex dictionary (ALL_PATTERNS).

Parameters you can tweak:
- `ALL_PATTERNS`: dictionary of regex patterns you maintain.
- `sensitive_categories`: semantic categories to detect with SpaCy.
- `threshold` in SemanticSimilarityRecognizer: controls how strict the similarity detection is.
"""

from presidio_analyzer import AnalyzerEngine, RecognizerRegistry, PatternRecognizer, RecognizerResult, EntityRecognizer
from presidio_anonymizer import AnonymizerEngine
import spacy
import re

# ----------------------
# Load SpaCy model
# ----------------------
# Use a medium or large model for better word vectors.
nlp = spacy.load("en_core_web_md")

# ----------------------
# Your custom regex dictionary
# ----------------------
# Example: add your predefined regex constants (IBAN_REGEX, etc.)
ALL_PATTERNS = {
    # C4 - Critical Risk
    'iban': r'[A-Z]{2}[0-9]{2}[A-Z0-9]{1,30}',
    'credit_card': r'\b(?:\d[ -]*?){13,16}\b',
    'social_security': r'\b\d{3}-\d{2}-\d{4}\b',
    'pin': r'\b\d{4}\b',
    'cvv': r'\b\d{3,4}\b',
    'transaction': r'TRX[0-9]{6,}',
    'phone': r'\+?[0-9]{7,15}',

    # C3 - High Risk
    'email': r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+',
    'customer_number': r'CUST[0-9]{6,}',
    'date_of_birth': r'\b\d{2}/\d{2}/\d{4}\b',
    'belgian_id': r'\b\d{11}\b',
    'address': r'\d+\s+\w+\s+(Street|St|Ave|Road|Rd)\b',
    'name': r'[A-Z][a-z]+\s[A-Z][a-z]+',
    'postal_code': r'\b\d{4,5}\b',
    'citizenship': r'Belgian|French|German|Dutch|Spanish|Italian',
}

# ----------------------
# Define sensitive categories for semantic similarity
# ----------------------
# Extend these with terms that should trigger anonymization even if not exact match.
sensitive_categories = {
    "RELIGION": ["christian", "muslim", "jewish", "hindu", "buddhist", "atheist"],
    "ETHNICITY": ["asian", "african", "latino", "caucasian", "arab", "indigenous"],
    "SEXUAL_ORIENTATION": ["gay", "lesbian", "bisexual", "transgender", "queer", "lgbtq"],
}

# Precompute embeddings for sensitive terms
category_embeddings = {
    category: [nlp(term)[0].vector for term in terms]
    for category, terms in sensitive_categories.items()
}

# ----------------------
# Custom SpaCy-based semantic similarity recognizer
# ----------------------
class SemanticSimilarityRecognizer(EntityRecognizer):
    """
    A custom Presidio Recognizer that uses SpaCy embeddings to detect sensitive words
    related to categories like religion, ethnicity, or sexual orientation.
    """

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
# Setup Presidio Analyzer with regex + semantic recognizers
# ----------------------
registry = RecognizerRegistry()

# Add all regex patterns as recognizers automatically
for entity, pattern in ALL_PATTERNS.items():
    recognizer = PatternRecognizer(
        supported_entity=entity.upper(),
        patterns=[{"name": entity, "pattern": pattern, "score": 0.8}],
    )
    registry.add_recognizer(recognizer)

# Add semantic similarity recognizer
semantic_recognizer = SemanticSimilarityRecognizer()
registry.add_recognizer(semantic_recognizer)

# Initialize Analyzer and Anonymizer
analyzer = AnalyzerEngine(registry=registry, supported_languages=["en"])
anonymizer = AnonymizerEngine()

# ----------------------
# Combined analysis function
# ----------------------
def analyze_and_anonymize(text):
    """
    Run analysis and anonymization on input text.

    Args:
        text (str): Input text to analyze.

    Returns:
        str: Anonymized text.
    """

    # Analyzer runs all registered recognizers (regex + semantic)
    results = analyzer.analyze(text=text, language="en")

    # Anonymize all detected entities
    anonymized_text = anonymizer.anonymize(text=text, analyzer_results=results)
    return anonymized_text

# ----------------------
# Example usage
# ----------------------
if __name__ == "__main__":
    sample_text = "John is a Christian engineer. Contact him at john.doe@email.com"
    result = analyze_and_anonymize(sample_text)
    print("Original:", sample_text)
    print("Anonymized:", result.text)
