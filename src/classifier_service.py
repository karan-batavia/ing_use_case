import re
import joblib
from pathlib import Path

class ScrubEngine:
    """Simple wrapper to load and use the sensitivity classifier."""
    def __init__(self, sk_model_path="sensitivity_classifier.joblib", hf_model_dir=None):
        self.model_path = Path(sk_model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        data = joblib.load(self.model_path)
        self.pipeline = data["pipeline"]
        self.labels = data["labels"]
        self.redactor = SimpleRedactor()

    def analyze(self, text: str):
        """Classify + redact text."""
        label = self.pipeline.predict([text])[0]
        probs = dict(zip(self.labels, self.pipeline.predict_proba([text])[0]))
        redacted = self.redactor.redact_text(text)
        return {
            "label": label,
            "probs": probs,
            "redacted_text": redacted["text"],
            "mapping": redacted["mapping"],
        }


class SimpleRedactor:
    """Lightweight regex-based redactor."""
    import re
    PATTERNS = {
        "EMAIL": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
        "PHONE": r"(?:\+\d{1,3}\s?)?(?:\d[\s-]?){9,}",
        "IBAN": r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b",
        "NATIONAL_ID": r"\bID[:\s-]?[A-Z0-9]{6,}\b",
        "SSN": r"\b\d{2}[.\-/]\d{2}[.\-/]\d{2}[- ]?\d{2,4}(?:[.\-]?\d{0,2})?\b",
    }
    COMPILED = {k: re.compile(v, re.IGNORECASE) for k, v in PATTERNS.items()}

    def redact_text(self, text: str):
        out = text
        mapping = []
        for key, pattern in self.COMPILED.items():
            for m in pattern.finditer(out):
                placeholder = f"[{key}_001]"
                mapping.append({
                    "type": key,
                    "original": m.group(),
                    "placeholder": placeholder
                })
                out = out.replace(m.group(), placeholder)
        return {"text": out, "mapping": mapping}