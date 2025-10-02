# recognizers.py

import spacy
from presidio_analyzer import EntityRecognizer, RecognizerResult

# Load SpaCy once per language
nlp_en = spacy.load("en_core_web_md")
nlp_fr = spacy.load("fr_core_news_md")
nlp_nl = spacy.load("nl_core_news_md")

# ----------------------
# Semantic Similarity Recognizer
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

class SemanticSimilarityRecognizer(EntityRecognizer):
    def __init__(self, threshold=0.7):
        super().__init__(supported_entities=list(sensitive_categories.keys()), supported_language="en")
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
# Heuristic Recognizer
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
# SpaCy Person Recognizer
# ----------------------
class SpacyPersonRecognizer(EntityRecognizer):
    def __init__(self, nlp_model, lang_code):
        super().__init__(supported_entities=["PERSON"], supported_language=lang_code)
        self.nlp = nlp_model

    def analyze(self, text, entities, nlp_artifacts=None):
        results = []
        doc = self.nlp(text)
        for ent in doc.ents:
            if ent.label_ in ["PER", "PERSON"]:
                results.append(
                    RecognizerResult(
                        entity_type="PERSON",
                        start=ent.start_char,
                        end=ent.end_char,
                        score=0.85,
                    )
                )
        return results

# Initialize PERSON recognizers for multiple languages
PERSON_RECOGNIZERS = [
    SpacyPersonRecognizer(nlp_en, "en"),
    SpacyPersonRecognizer(nlp_fr, "fr"),
    SpacyPersonRecognizer(nlp_nl, "nl"),
]
