#!/usr/bin/env python3
"""
Hybrid Sensitivity Classifier (rules + ML) with multi-format support.

Supports: TXT, CSV, PDF, DOCX, and images (PNG, JPG, JPEG)
- DATA_PATH can be set via env or --data argument
- MODEL can be: logreg | rf (roberta is a stub that raises NotImplementedError)
- Robust tokenization (unicode-friendly) + Hashing fallback to avoid 'empty vocabulary' errors
- Saves trained pipeline to sensitivity_classifier.joblib in the script directory
"""

import os
import re
import sys
import json
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
import joblib

import spacy
from langdetect import detect
import unicodedata

class SpacyPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, lang_map=None):
        # Default language model map
        self.lang_map = lang_map or {
            "en": spacy.load("en_core_web_sm"),
            "fr": spacy.load("fr_core_news_sm"),
            "nl": spacy.load("nl_core_news_sm"),
        }

    def detect_lang(self, text: str) -> str:
        try:
            return detect(text)
        except:
            return "en"

    def normalize(self, text: str) -> str:
        return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8').lower()

    def preprocess(self, text: str, lang: str) -> str:
        nlp = self.lang_map.get(lang, self.lang_map["en"])
        doc = nlp(text)
        tokens = [token.lemma_.lower() for token in doc if not token.is_stop and token.is_alpha]
        return " ".join(tokens)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        processed = []
        for text in X:
            lang = self.detect_lang(text)
            cleaned = self.normalize(text)
            processed.append(self.preprocess(cleaned, lang))
        return processed

# File parsing libraries
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("Warning: pandas not installed. CSV support disabled.")

try:
    import PyPDF2
    HAS_PDF = True
except ImportError:
    HAS_PDF = False
    print("Warning: PyPDF2 not installed. PDF support disabled.")

try:
    from PIL import Image
    import pytesseract
    HAS_OCR = True
except ImportError:
    HAS_OCR = False
    print("Warning: PIL/pytesseract not installed. Image OCR support disabled.")

try:
    from docx import Document
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False
    print("Warning: python-docx not installed. DOCX support disabled.")

# --- Integrations: your regex modules ---
EXT_REGEX_OK = False
EXT_C12_OK = False
try:
    # regex_queries.py provides: ALL_PATTERNS (compiled re.Patterns)
    from .preprocessing.regex_queries import ALL_PATTERNS as RQ_ALL
    EXT_REGEX_OK = isinstance(RQ_ALL, dict) and len(RQ_ALL) > 0
except Exception as e:
    EXT_REGEX_OK = False
    print(f"Failed to import regex_queries.py: {e}")

try:
    # regex.py provides: extract_c1_and_c2(text_filepath, data_dir) and transform_text(text, c1, c2)
    import preprocessing.regex as c12mod  
    EXT_C12_OK = hasattr(c12mod, "extract_c1_and_c2") and hasattr(c12mod, "transform_text")
except Exception:
    EXT_C12_OK = False

# Feature flags (env)
USE_REGEX_QUERIES = os.environ.get("USE_REGEX_QUERIES", "1") == "1"
USE_C12_NORMALIZE = os.environ.get("USE_C12_NORMALIZE", "0") == "1"
C12_DATA_DIR = os.environ.get("C12_DATA_DIR", "")

# --------------------------
# Config (env + sensible defaults)
# --------------------------
DEFAULT_DATA = "raw_prompts.txt"
DATA_PATH = os.environ.get("DATA_PATH", DEFAULT_DATA)
MODEL_CHOICE = os.environ.get("MODEL", "logreg").lower()
RANDOM_STATE = int(os.environ.get("SEED", "42"))
TEST_SIZE = float(os.environ.get("TEST_SIZE", "0.2"))
NGRAM_MAX = int(os.environ.get("NGRAM_MAX", "2"))

_env_min_df = os.environ.get("MIN_DF", "1")
MIN_DF = int(_env_min_df) if _env_min_df.isdigit() else 1
MAX_DF = float(os.environ.get("MAX_DF", "1.0"))

CSV_TEXT_COLUMN = os.environ.get("CSV_TEXT_COLUMN", None)

# --------------------------
# Optional user regex patterns
# --------------------------
USER_REGEX_OK = False
try:
    import regex as user_regex  
    USER_REGEX_OK = hasattr(user_regex, "PATTERNS")
except Exception:
    USER_REGEX_OK = False

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

def _to_compiled(v) -> re.Pattern:
    if isinstance(v, re.Pattern):
        return re.compile(v.pattern, re.IGNORECASE)
    return re.compile(str(v), re.IGNORECASE)

def get_patterns() -> Dict[str, re.Pattern]:
    merged: Dict[str, re.Pattern] = {}
    for k, pat in FALLBACK_PATTERNS.items():
        merged[k] = _to_compiled(pat)

    if USER_REGEX_OK and isinstance(user_regex.PATTERNS, dict):
        for k, pat in user_regex.PATTERNS.items():
            k_upper = k.upper()
            if k_upper in merged:
                print(f"Overriding pattern for {k_upper} from user_regex")
            merged[k_upper] = _to_compiled(pat)

    if USE_REGEX_QUERIES and EXT_REGEX_OK:
        for k, pat in RQ_ALL.items():
            k_upper = k.upper()
            if k_upper in merged:
                print(f"Overriding pattern for {k_upper} from regex_queries")
            merged[k_upper] = _to_compiled(pat)

    return merged

PATTERNS: Dict[str, re.Pattern] = get_patterns()

# --------------------------
# Labels + heuristics
# --------------------------
LABELS = ["C1", "C2", "C3", "C4"]

def heuristic_label(text: str) -> str:
    t = text.lower()

    if USE_C12_NORMALIZE and ("<" in text and ">" in text):
        if any(tag in t for tag in ["<iban", "<credit_card", "<social_security", "<pin", "<cvv", "<transaction", "<phone"]):
            return "C4"
        if any(tag in t for tag in ["<email", "<customer_number", "<date_of_birth", "<address", "<belgian_id", "<employee_id", "<contract_number"]):
            return "C3"
    
    c4_hits = any(PATTERNS[k].search(text) for k in ["SSN_LIKE", "IBAN", "NATIONAL_ID", "BIOMETRIC"])
    email_hit = PATTERNS["EMAIL"].search(text) is not None
    amount_hit = PATTERNS["AMOUNT"].search(text) is not None
    c4_words = any(w in t for w in [
        "credit score", "cote de crédit", "cotes de crédit", "kredietscore", "kredietscores",
        "income", "revenu", "revenus", "inkomen", "inkomens",
        "account balance", "solde du compte", "soldes des comptes", "rekeningsaldo", "rekeningsaldi",
        "pin", "PIN", "pincode", "pincodes",
        "national id", "carte d’identité", "identiteitskaart", "identiteitskaarten",
        "corpkeys", "clé d’entreprise", "bedrijfsleutel", "bedrijfsleutels",
        "biometric", "biométrique", "biométriques", "biometrisch", "biometrische",
        "sepa",
        "wire transfer", "virement", "virements", "overschrijving", "overschrijvingen",
        "cheque", "cheques",
        "direct debit", "prélèvement automatique", "prélèvements automatiques", "domiciliëring", "domiciliëringen"
    ])
    if c4_hits or (email_hit and amount_hit) or c4_words:
        return "C4"

    c3_words = any(w in t for w in [
        "overeenkomst", "overeenkomsten", "supplier", "fournisseur", "fournisseuse", "leverancier", "leveranciers", "customer",
        "client", "clientes", "klant", "klanten", "standing order", "ordre permanent", "ordres permanents", "doorlopende opdracht", 
        "doorlopende opdrachten", "payment order", "ordre de paiement", "ordres de paiement", "betaalopdracht", "betaalopdrachten", 
        "deposit",  "dépôt", "storting", "stortingen", "overdraft", "découvert", "découverts", "roodstand", "roodstanden", "line of credit",
        "ligne de crédit", "lignes de crédit", "kredietlijn", "kredietlijnen", "bank guarantee", "garantie bancaire", "garanties bancaires",
        "bankgarantie", "bankgaranties", "letter of credit", "lettre de crédit", "lettres de crédit", "kredietbrief", "kredietbrieven", "invoice", 
        "facture", "factures", "factuur", "facturen"
    ])
    if c3_words or amount_hit:
        return "C3"

    c2_words = any(w in t for w in [
        "policy","guideline","standard","sop","governance","raci","owner","approver",
        "review cycle","deprecated","retired","under review","internal","digest","reminder","notice"
    ])
    if c2_words:
        return "C2"

    c1_words = any(w in t for w in [
    "annual report", "rapport annuel", "rapports annuels", "jaarverslag", "jaarverslagen", "pillar 3",
    "pilier 3", "piliers 3", "pijler 3", "pijlers 3", "press", "presse", "pers", "newsroom", "salle de presse",
    "salles de presse", "perskamer", "perskamers", "full year results", "résultats annuels", "jaarresultaten", 
    "half year results", "résultats semestriels", "halfjaarresultaten", "public disclosures", "divulgation publique", 
    "divulgations publiques", "openbare bekendmaking", "openbare bekendmakingen", "investor", "investisseur", "investisseuse",
    "investisseurs", "investisseuses", "investeerder", "investeerders", "analyst", "analyste", "analystes", "analist", "analisten", 
    "linkedin", "press-room", "salle de presse", "salles de presse", "persruimte", "persruimtes", "pdf", "xlsx", "report", "rapport",
    "rapports", "verslag", "verslagen"    
    ])
    if c1_words:
        return "C1"

    return "C2"

# --------------------------
# File parsers
# --------------------------
def parse_txt(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines

def parse_csv(path: Path) -> List[str]:
    if not HAS_PANDAS:
        raise RuntimeError("pandas is required for CSV parsing. Install: pip install pandas")
    
    df = pd.read_csv(path)
    texts = []
    
    if CSV_TEXT_COLUMN and CSV_TEXT_COLUMN in df.columns:
        texts = df[CSV_TEXT_COLUMN].astype(str).tolist()
    else:
        for _, row in df.iterrows():
            combined = " ".join(str(val) for val in row.values if pd.notna(val))
            if combined.strip():
                texts.append(combined.strip())
    
    return texts

def parse_pdf(path: Path) -> List[str]:
    if not HAS_PDF:
        raise RuntimeError("PyPDF2 is required for PDF parsing. Install: pip install PyPDF2")
    
    texts = []
    with path.open("rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text = page.extract_text()
            if text and text.strip():
                lines = [line.strip() for line in text.split('\n') if line.strip()]
                texts.extend(lines)
    
    return texts

def parse_docx(path: Path) -> List[str]:
    if not HAS_DOCX:
        raise RuntimeError("python-docx is required for DOCX parsing. Install: pip install python-docx")
    
    doc = Document(str(path))
    texts = []
    
    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            texts.append(text)
    
    for table in doc.tables:
        for row in table.rows:
            row_text = " ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
            if row_text:
                texts.append(row_text)
    
    return texts

def parse_image(path: Path) -> List[str]:
    if not HAS_OCR:
        raise RuntimeError("PIL and pytesseract are required for image OCR. "
                         "Install: pip install Pillow pytesseract")
    
    img = Image.open(path)
    text = pytesseract.image_to_string(img)
    
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    return lines

def parse_file(path: Path) -> List[str]:
    ext = path.suffix.lower()
    
    if ext == ".txt":
        return parse_txt(path)
    elif ext == ".csv":
        return parse_csv(path)
    elif ext == ".pdf":
        return parse_pdf(path)
    elif ext in [".docx", ".doc"]:
        return parse_docx(path)
    elif ext in [".png", ".jpg", ".jpeg", ".tiff", ".bmp"]:
        return parse_image(path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

# --------------------------
# Data loader
# --------------------------
def _c12_transform_text(text: str) -> str:
    if not (USE_C12_NORMALIZE and EXT_C12_OK):
        return text

    from tempfile import NamedTemporaryFile
    try:
        with NamedTemporaryFile("w+", suffix=".txt", delete=True, encoding="utf-8") as tmp:
            tmp.write(text)
            tmp.flush()
            data_dir = C12_DATA_DIR if C12_DATA_DIR else os.getcwd()
            res = c12mod.extract_c1_and_c2(tmp.name, data_dir)
            return c12mod.transform_text(text, res.get("c1", {}), res.get("c2", {}))
    except Exception:
        return text

def load_dataset(path: str) -> Tuple[List[str], List[str]]:
    p = Path(path).resolve()
    if not p.exists():
        alternatives = [
            Path(path),
            Path.cwd() / path,
            Path(__file__).parent / path,
        ]
        for alt in alternatives:
            if alt.exists():
                p = alt.resolve()
                break
        else:
            raise FileNotFoundError(
                f"Dataset not found at: {path}\n"
                f"Tried absolute path: {Path(path).resolve()}\n"
                f"Current directory: {Path.cwd()}\n"
                f"Pass correct path via DATA_PATH env or --data argument."
            )
    
    print(f"Loading dataset from: {p} (format: {p.suffix})")
    texts = parse_file(p)
    
    if not texts:
        raise ValueError("Dataset is empty after parsing.")
    
    labels = [heuristic_label(text) for text in texts]
    
    print(f"Loaded {len(texts)} text samples")
    return texts, labels

# --------------------------
# Rule-based featureizer
# --------------------------
class RuleFeatureizer(BaseEstimator, TransformerMixin):
    def __init__(self, pattern_keys: Optional[List[str]] = None):
        self.pattern_keys = pattern_keys or list(PATTERNS.keys())

    def fit(self, X: List[str], y=None):
        return self

    def transform(self, X: List[str]) -> Any:
        import numpy as np
        feats = []
        for text in X:
            row = []
            for k in self.pattern_keys:
                p = PATTERNS[k]
                matches = p.findall(text)
                row.append(int(bool(matches)))
                row.append(len(matches))
            
            def has_any(words):
                tl = text.lower()
                return int(any(w in tl for w in words))
            
            row.extend([
                has_any(["credit score","income","account balance","masked pin","biometric"]),
                has_any(["agreement","supplier","customer","standing order","payment order","overdraft"]),
                has_any(["annual report","pillar 3","press","newsroom","full year results","investor"]),
                has_any(["policy","guideline","standard","sop","governance","raci","deprecated","retired"]),
            ])
            feats.append(row)
        return np.array(feats, dtype=float)

# --------------------------
# Feature builders
# --------------------------
def build_features(use_hash: bool = False) -> FeatureUnion:
    rules = RuleFeatureizer()
    spacy_prep = SpacyPreprocessor()

    if use_hash:
        bow = HashingVectorizer(
            n_features=2**16,
            alternate_sign=False,
            lowercase=False,
            token_pattern=r"(?u)\b\w+\b",
        )
        return FeatureUnion([
            ("bow", Pipeline([("spacy", spacy_prep), ("vectorizer", bow)])),
            ("rules", rules)
        ])
    
    tfidf = TfidfVectorizer(
        ngram_range=(1, NGRAM_MAX),
        min_df=MIN_DF,
        max_df=MAX_DF,
        lowercase=False,
        token_pattern=r"(?u)\b\w+\b",
    )
    return FeatureUnion([
        ("tfidf", Pipeline([("spacy", spacy_prep), ("vectorizer", tfidf)])),
        ("rules", rules)
    ])

# --------------------------
# Classifier chooser
# --------------------------
def build_classifier(name: str):
    if name == "logreg":
        return LogisticRegression(max_iter=300, solver="lbfgs", multi_class="auto", random_state=RANDOM_STATE)
    if name == "rf":
        return RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, class_weight="balanced")
    if name == "roberta":
        raise NotImplementedError("RoBERTa fine-tune is not enabled in this simple setup. Use MODEL=logreg or MODEL=rf.")
    raise ValueError(f"Unknown MODEL: {name}")

# --------------------------
# Train / Eval
# --------------------------
@dataclass
class TrainResult:
    report: str
    conf_mat: List[List[int]]
    model_path: str

def train_and_eval(data_path: str) -> TrainResult:
    texts, labels = load_dataset(data_path)
    
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=labels
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=None
        )

    clf = build_classifier(MODEL_CHOICE)

    use_hash = False
    try:
        pipe = Pipeline([("features", build_features(use_hash)), ("clf", clf)])
        pipe.fit(X_train, y_train)
    except ValueError as e:
        msg = str(e).lower()
        if "empty vocabulary" in msg or "no terms remain" in msg:
            use_hash = True
            pipe = Pipeline([("features", build_features(use_hash)), ("clf", clf)])
            pipe.fit(X_train, y_train)
        else:
            raise

    y_pred = pipe.predict(X_test)
    report = classification_report(y_test, y_pred, digits=3, labels=LABELS, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=LABELS).tolist()

    out_path = str((Path(__file__).parent / "sensitivity_classifier.joblib").resolve())
    joblib.dump({"pipeline": pipe, "labels": LABELS}, out_path)

    return TrainResult(report=report, conf_mat=cm, model_path=out_path)

# --------------------------
# CLI
# --------------------------
def parse_args(argv: List[str]):
    data_arg = None
    if "--data" in argv:
        i = argv.index("--data")
        if i + 1 < len(argv):
            data_arg = argv[i + 1]
    return data_arg

def main(argv: List[str] = None):
    argv = argv or sys.argv[1:]
    data_arg = parse_args(argv)
    data_path = data_arg or DATA_PATH

    if "--train" in argv or len(argv) == 0:
        res = train_and_eval(data_path)
        print("\n=== Sensitivity Classifier (Hybrid: rules + TF-IDF + ML) ===")
        print(f"Model: {MODEL_CHOICE}")
        print(f"Data:  {data_path}")
        print("\n--- Classification Report ---")
        print(res.report)
        print("\n--- Confusion Matrix [rows=true, cols=pred] (order C1,C2,C3,C4) ---")
        print(json.dumps(res.conf_mat))
        print(f"\nModel saved to: {res.model_path}")
    else:
        print("Usage:")
        print("  python ml_setup.py --train --data /path/to/file.[txt|csv|pdf|docx|png|jpg]")
        print("  DATA_PATH=/path/to/file.csv MODEL=logreg python ml_setup.py --train")
        print("\nSupported formats: TXT, CSV, PDF, DOCX, PNG, JPG, JPEG")
        print("\nEnvironment variables:")
        print("  DATA_PATH: Path to input file")
        print("  MODEL: logreg (default) | rf")
        print("  CSV_TEXT_COLUMN: Specific column name for CSV (default: concatenate all)")

if __name__ == "__main__":
    main()