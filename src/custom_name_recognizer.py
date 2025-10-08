from presidio_analyzer import EntityRecognizer, RecognizerResult
import spacy

nlp = spacy.load("en_core_web_lg")  # use a large model for better NER accuracy

# Words that look like titles, reports, events, etc. — not names
NON_PERSON_TOKENS = {
    "year", "report", "results", "disclosure", "press", "policy",
    "guideline", "overview", "meeting", "summary", "board", "annual",
    "statement", "review", "update", "analysis", "project", "agreement"
}

class TruePersonRecognizer(EntityRecognizer):
    """
    Custom recognizer for PERSON entities that filters out 'false names'
    like 'Full Year Results' or 'Annual Report'.
    """

    def __init__(self, supported_entities=None, language="en"):
        super().__init__(supported_entities or ["PERSON"], language)

    def analyze(self, text, entities, nlp_artifacts=None):
        results = []
        doc = nlp_artifacts if nlp_artifacts else nlp(text)

        for ent in doc.ents:
            if ent.label_ != "PERSON":
                continue

            token_set = {t.text.lower() for t in ent if t.is_alpha}
            # if it's made of forbidden words → skip
            if token_set & NON_PERSON_TOKENS:
                continue

            # if it's too long or short → skip
            if len(ent.text.split()) > 3 or len(ent.text) < 3:
                continue

            # heuristic: must contain at least one capitalized word that is not generic
            if not any(t[0].isupper() for t in ent.text.split()):
                continue

            results.append(
                RecognizerResult(
                    entity_type="PERSON",
                    start=ent.start_char,
                    end=ent.end_char,
                    score=float(ent.label_ == "PERSON")
                )
            )
        return results