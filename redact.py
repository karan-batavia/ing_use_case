
#!/usr/bin/env python3
import re
import os
import json
import csv
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

# Optional user regex file
USER_REGEX_OK = False
try:
    import regex as user_regex  
    USER_REGEX_OK = hasattr(user_regex, "PATTERNS")
except Exception:
    USER_REGEX_OK = False

FALLBACK_PATTERNS: Dict[str, str] = {
    "EMAIL": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
    "PHONE_EU": r"(?:\+\d{1,3}\s?)?(?:\d[\s-]?){9,}",
    "SSN_LIKE": r"\b\d{2}[.\-/]\d{2}[.\-/]\d{2}[- ]?\d{2,4}[.\-]?\d{0,2}\b|\b\d{6}[- ]?\d{2,4}[.\-]?\d{0,2}\b",
    "IBAN": r"\b[A-Z]{2}\d{2}[ ]?(?:[A-Z0-9]{3,4}[ ]?){3,7}\b",
    "ACCOUNT_NUM": r"\b(?:acct|account)\s*\d{3,}\b",
    "AMOUNT": r"(?:USD|EUR|GBP|€|\$|£)\s?\d{1,3}(?:[, \u00A0]\d{3})*(?:\.\d{2})?",
    "DOB": r"\b\d{4}-\d{2}-\d{2}\b",
    "NATIONAL_ID": r"\b(?:national (?:id|number)|id)[:\s-]*[A-Z0-9.\-]{6,}\b",
    "BIOMETRIC": r"\b(FaceID|fingerprint|iris|biometric)\b",
    "IBAN_PLAIN": r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b",
}

def _compile_patterns() -> Dict[str, re.Pattern]:
    base = FALLBACK_PATTERNS.copy()
    if USER_REGEX_OK and isinstance(user_regex.PATTERNS, dict):
        base.update(user_regex.PATTERNS)  # user overrides or extends
    return {k: re.compile(v, re.IGNORECASE) for k, v in base.items()}

EXPLANATIONS: Dict[str, str] = {
    "EMAIL": "Email addresses are personally identifiable information (PII).",
    "PHONE_EU": "Phone numbers are PII and can identify or contact an individual.",
    "SSN_LIKE": "National/SSN-like identifiers are highly sensitive government-issued IDs.",
    "IBAN": "IBAN corresponds to a bank account identifier and is financial PII.",
    "IBAN_PLAIN": "IBAN corresponds to a bank account identifier and is financial PII.",
    "ACCOUNT_NUM": "Account numbers relate to financial PII and should be masked.",
    "AMOUNT": "Monetary amounts may be business-sensitive when linked to people or transactions.",
    "DOB": "Date of birth is personal data that can identify an individual.",
    "NATIONAL_ID": "National ID/number is a highly sensitive personal identifier.",
    "BIOMETRIC": "Biometric references relate to sensitive authentication factors.",
}

TYPE_PRIORITY = [
    "SSN_LIKE", "NATIONAL_ID", "IBAN", "IBAN_PLAIN",
    "ACCOUNT_NUM", "EMAIL", "PHONE_EU", "DOB", "AMOUNT", "BIOMETRIC"
]

@dataclass
class MatchItem:
    type: str
    start: int
    end: int
    text: str

class Redactor:
    def __init__(self, patterns: Optional[Dict[str, re.Pattern]] = None):
        self.patterns = patterns or _compile_patterns()
        self.counters: Dict[str, int] = {}

    def _placeholder(self, typ: str) -> str:
        n = self.counters.get(typ, 0) + 1
        self.counters[typ] = n
        return f"[{typ}_{n:03d}]"

    def find_entities(self, text: str) -> List[MatchItem]:
        found: List[MatchItem] = []
        for typ, pat in self.patterns.items():
            for m in pat.finditer(text):
                found.append(MatchItem(type=typ, start=m.start(), end=m.end(), text=m.group(0)))
        if not found:
            return []

        def prio(mi: MatchItem) -> Tuple[int, int, int]:
            length = mi.end - mi.start
            pidx = TYPE_PRIORITY.index(mi.type) if mi.type in TYPE_PRIORITY else len(TYPE_PRIORITY)
            return (-length, pidx, mi.start)
        found.sort(key=prio)

        selected: List[MatchItem] = []
        occupied = []
        for mi in found:
            overlaps = any(not (mi.end <= s or mi.start >= e) for (s, e) in occupied)
            if not overlaps:
                selected.append(mi)
                occupied.append((mi.start, mi.end))

        selected.sort(key=lambda x: x.start)
        return selected

    def redact_text(self, text: str) -> Dict[str, Any]:
        entities = self.find_entities(text)
        if not entities:
            return {"text": text, "mapping": [], "index": {}}

        out_chars = []
        last = 0
        mapping: List[Dict[str, Any]] = []
        index: Dict[str, str] = {}

        for mi in entities:
            out_chars.append(text[last:mi.start])
            ph = self._placeholder(mi.type)
            out_chars.append(ph)
            item = {
                "type": mi.type,
                "placeholder": ph,
                "original": mi.text,
                "span": [mi.start, mi.end],
                "explanation": EXPLANATIONS.get(mi.type, "Redacted sensitive entity."),
            }
            mapping.append(item)
            index[ph] = mi.text
            last = mi.end

        out_chars.append(text[last:])
        redacted = "".join(out_chars)

        return {"text": redacted, "mapping": mapping, "index": index}

    def redact_lines(self, lines: List[str]) -> List[Dict[str, Any]]:
        return [self.redact_text(ln) for ln in lines]

# def _write_outputs(results, out_path, map_path, csv_path):
#     if out_path:
#         with open(out_path, "w", encoding="utf-8") as f:
#             for r in results:
#                 f.write(r["text"] + "\n")
#     if map_path:
#         with open(map_path, "w", encoding="utf-8") as f:
#             json.dump(results, f, ensure_ascii=False, indent=2)
#     if csv_path:
#         with open(csv_path, "w", newline="", encoding="utf-8") as f:
#             w = csv.writer(f)
#             w.writerow(["line_idx","type","placeholder","original","span_start","span_end","explanation"])
#             for i, r in enumerate(results):
#                 for m in r["mapping"]:
#                     w.writerow([i, m["type"], m["placeholder"], m["original"], m["span"][0], m["span"][1], m["explanation"]])

def _ensure_parent(path: str):
    import os
    if not path:
        return
    parent = os.path.dirname(path)
    if parent:  # only if there is a parent dir component
        os.makedirs(parent, exist_ok=True)

def _write_outputs(results, out_path, map_path, csv_path):
    # ensure folders exist
    _ensure_parent(out_path)
    _ensure_parent(map_path)
    _ensure_parent(csv_path)

    # Redacted text file
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(r["text"] + "\n")

    # JSON mapping (per-line objects)
    if map_path:
        import json
        with open(map_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    # Flat CSV (one row per replacement)
    if csv_path:
        import csv
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["line_idx","type","placeholder","original","span_start","span_end","explanation"])
            for i, r in enumerate(results):
                for m in r["mapping"]:
                    w.writerow([i, m["type"], m["placeholder"], m["original"], m["span"][0], m["span"][1], m["explanation"]])
def main():
    import argparse, pathlib
    ap = argparse.ArgumentParser(description="Redact entities with placeholders and keep a mapping.")
    ap.add_argument("--in", dest="infile", required=True, help="Input text file (one record per line)")
    ap.add_argument("--out", dest="outfile", required=True, help="Output redacted text file")
    ap.add_argument("--map", dest="mapfile", default=None, help="JSON with redaction mappings")
    ap.add_argument("--csv", dest="csvfile", default=None, help="CSV with flattened mappings")
    args = ap.parse_args()

    inp = pathlib.Path(args.infile)
    if not inp.exists():
        ap.error(f"Input file not found: {inp}")

    lines = [ln.rstrip("\n") for ln in inp.read_text(encoding="utf-8").splitlines()]

    red = Redactor()
    results = red.redact_lines(lines)

    _write_outputs(results, args.outfile, args.mapfile, args.csvfile)

if __name__ == "__main__":
    main()
