# presidio_pipeline.py

from presidio_analyzer import AnalyzerEngine, RecognizerRegistry, PatternRecognizer
from presidio_anonymizer import AnonymizerEngine
import joblib
from pathlib import Path

# Import regexes
from regex_queries import ALL_PATTERNS
from patterns_shared import PATTERNS as ML_PATTERNS

# Import custom recognizers
from recognizers import SemanticSimilarityRecognizer, HeuristicRecognizer, PERSON_RECOGNIZERS
from entity_mapping import ENTITY_SENSITIVITY

# ----------------------
# Setup registry
# ----------------------
registry = RecognizerRegistry()

# Add regex recognizers (ALL_PATTERNS + ML_PATTERNS)
for source in (ALL_PATTERNS, ML_PATTERNS):
    for entity, pattern in source.items():
        recognizer = PatternRecognizer(
            supported_entity=entity.upper(),
            patterns=[{"name": entity, "pattern": pattern, "score": 0.8}],
        )
        registry.add_recognizer(recognizer)

# Add semantic + heuristic + PERSON recognizers
registry.add_recognizer(SemanticSimilarityRecognizer())
registry.add_recognizer(HeuristicRecognizer())
for person_rec in PERSON_RECOGNIZERS:
    registry.add_recognizer(person_rec)

# Initialize engines
analyzer = AnalyzerEngine(registry=registry, supported_languages=["en", "fr", "nl"])
anonymizer = AnonymizerEngine()

# Load ML classifier
MODEL_PATH = Path(__file__).parent / "sensitivity_classifier.joblib"
clf_bundle = joblib.load(MODEL_PATH)
classifier = clf_bundle["pipeline"]
labels = clf_bundle["labels"]

# ----------------------
# Combined pipeline
# ----------------------
def analyze_and_anonymize_with_classification(text: str, lang="en"):
    results = analyzer.analyze(text=text, language=lang)
    anonymized_text = anonymizer.anonymize(text=text, analyzer_results=results)

    # Map entities to sensitivity
    entities_with_levels = []
    for r in results:
        mapped_level = ENTITY_SENSITIVITY.get(r.entity_type, "UNKNOWN")
        e = r.to_dict()
        e["mapped_level"] = mapped_level
        entities_with_levels.append(e)

    sensitivity = classifier.predict([text])[0]

    return {
        "original": text,
        "anonymized": anonymized_text.text,
        "entities": entities_with_levels,
        "sensitivity": sensitivity,
    }
