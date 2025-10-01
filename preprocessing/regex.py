import os
import re
from collections import OrderedDict, defaultdict
from datetime import datetime
import pandas as pd


# ----------------------------
# Helpers
# ----------------------------

def normalize_dash(s: str) -> str:
    """Normalize hyphen/em dash to en dash."""
    return re.sub(r'[—-]', '–', s)


def to_iso_date(s: str) -> str:
    """Normalize a date string to YYYY-MM-DD when possible."""
    s = s.strip()
    # ISO?
    m = re.match(r'^(20\d{2}|19\d{2})-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])$', s)
    if m:
        return s
    # Textual: October 15, 2025
    try:
        return datetime.strptime(s, "%B %d, %Y").strftime("%Y-%m-%d")
    except Exception:
        pass
    # EU: 15/10/2025
    try:
        return datetime.strptime(s, "%d/%m/%Y").strftime("%Y-%m-%d")
    except Exception:
        pass
    return s  # fallback to raw


def iter_matches_in_order(text: str, pattern: str | re.Pattern):
    """Yield unique matches in first-appearance order (flatten tuples)."""
    if isinstance(pattern, str):
        pattern = re.compile(pattern, re.I)
    seen = OrderedDict()
    for m in pattern.finditer(text):
        val = m.group(0)
        if isinstance(val, tuple):
            val = "".join(val)
        if val not in seen:
            seen[val] = True
    return list(seen.keys())


def assign_numbered(label: str, values_in_order: list[str]) -> dict:
    """Return a dict of placeholder -> value, numbered by appearance.
       If only 1 value, no suffix (_1) to mirror your original convention.
    """
    out = {}
    if not values_in_order:
        return out
    if len(values_in_order) == 1:
        out[f"<{label}>"] = values_in_order[0]
        return out
    for i, v in enumerate(values_in_order, 1):
        out[f"<{label}_{i}>"] = v
    return out


def next_index_for_label(label: str, current: dict) -> int:
    """Compute the next numeric suffix for placeholders of a given label in current dict."""
    pattern = re.compile(rf"^<{re.escape(label)}(?:_(\d+))?>$")
    max_i = 0
    for k in current.keys():
        m = pattern.match(k)
        if m:
            if m.group(1):
                max_i = max(max_i, int(m.group(1)))
            else:
                max_i = max(max_i, 1)
    return max_i + 1 if max_i else 1


def add_values_from_files(label: str, values: list[str], target_dict: dict):
    """Add new values for a label from files, without disturbing existing numbering/order."""
    existing_values = set(
        v for k, v in target_dict.items()
        if k.startswith(f"<{label}>")
    )
    for v in values:
        if v in existing_values:
            continue
        idx = next_index_for_label(label, target_dict)
        key = f"<{label}>" if idx == 1 else f"<{label}_{idx}>"
        target_dict[key] = v


def normalize_column(col):
    clean = str(col).strip().replace(";", "_").replace(" ", "_")
    return f"<{clean}>"


# ----------------------------
# Date context classification
# ----------------------------

REVIEW_WORDS = ("review", "reviewed", "revalidation", "validated", "validation", "audit", "audited")
DEPLOY_WORDS = ("deploy", "deployed", "deployment", "rollout", "rolled out")
EXEC_WORDS = ("execute", "executed", "execution", "run", "ran", "runned", "last run")

DATE_ALLOWED_PLACEHOLDERS = {
    "DATE_1", "DATE_2", "Last_Reviewed_Date", "Deployment_Date", "Last_Executed_Date"
}

def classify_text_date_context(around_text: str) -> str | None:
    """Classify a date by nearby text context."""
    low = around_text.lower()
    if any(w in low for w in REVIEW_WORDS):
        return "Last_Reviewed_Date"
    if any(w in low for w in DEPLOY_WORDS):
        return "Deployment_Date"
    if any(w in low for w in EXEC_WORDS):
        return "Last_Executed_Date"
    return None


def classify_file_date_context(neighbor_label: str | None, neighbor_column: str | None, neighbor_value: str | None) -> str | None:
    """Classify a date extracted from files using neighbor info."""
    txt = " ".join([
        (neighbor_label or ""), (neighbor_column or ""), (neighbor_value or "")
    ]).lower()
    if any(w in txt for w in REVIEW_WORDS):
        return "Last_Reviewed_Date"
    if any(w in txt for w in DEPLOY_WORDS):
        return "Deployment_Date"
    if any(w in txt for w in EXEC_WORDS):
        return "Last_Executed_Date"
    return None


# ----------------------------
# Main extraction
# ----------------------------

def extract_c1_and_c2(text_filepath, data_dir):
    with open(text_filepath, 'r', encoding='utf-8') as f:
        text = f.read()

    # C1 (contextual) patterns
    c1_patterns = {
        "LINK": re.compile(r'(?:https?://|www\.)[^\s<>"\)\]]+', re.I),
        "YEAR_RANGE": re.compile(r'\b(19\d{2}|20[0-4]\d|2050)\s*[–—-]\s*(19\d{2}|20[0-4]\d|2050)\b'),
        "YEAR": re.compile(r'\b(19\d{2}|20[0-4]\d|2050)\b'),
        "DOCUMENT_TYPE": re.compile(
            r'\b(?:pillar\s*3\s*(?:risk\s*)?(?:framework|report|disclosures?)|'
            r'(?:[\w-]+\s+){1,2}results|'
            r'[\w-]+\s+report)\b',
            re.I
        ),
        "DATE_ISO": re.compile(r'\b(20\d{2}|19\d{2})-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])\b'),
        "DATE_TEXTUAL": re.compile(r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(0?[1-9]|[12]\d|3[01]),\s*(20\d{2}|19\d{2})\b'),
        "DATE_EU": re.compile(r'\b(0?[1-9]|[12]\d|3[01])/(0?[1-9]|1[0-2])/(20\d{2}|19\d{2})\b'),
    }

    # C2 (entities)
    partial_patterns = {
        "VM_Name": re.compile(r'\bVM-IAAS-\d+\b', re.I),
        "Network_interface_ID": re.compile(r'\bNIC-IAAS-\d+\b', re.I),
        "System_name": re.compile(r'\bSYS-IAAS-\d+\b', re.I),
        "Design_Document_Reference": re.compile(r'\bDOC(?:\s+reference)?\b', re.I),
        "Design_Document_Code": re.compile(r'\bDOC-\d{4}-Q[1-4]-[A-Z]+(?:-[A-Z]+)?\b', re.I),
        "Fees": re.compile(r'\bannual\s+fee\b', re.I),
        "Version": re.compile(r'\bv\d+(?:\.\d+){1,2}\b', re.I),  # vX.Y or vX.Y.Z
        "e-mail": re.compile(r'\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b', re.I),
        "Phone_Number": re.compile(r'\+32(?:\s?\d){8,10}\b', re.I),
        "Frequency": re.compile(r'\b(quarterly|monthly|annual|yearly)\b', re.I),
        "Product_Code": re.compile(r'\b[A-Z]{2,}-\d{3,}\b'),
    }

    c1, c2 = {}, {}
    used_years_in_ranges = set()
    meta = {"dates": []}  # date-file-column neighbor context
    date_assignments = {}  # raw date string -> one of allowed placeholders

    # ---------- C1 Extraction (from text) ----------
    # YEAR_RANGE first (in order)
    yr_matches = []
    for m in c1_patterns["YEAR_RANGE"].finditer(text):
        a, b = m.group(1), m.group(2)
        yr_matches.append((a, b))

    if len(yr_matches) == 1:
        key = "<YEAR_RANGE>"
        value = normalize_dash("–".join(yr_matches[0]))
        c1[key] = value.strip()
        used_years_in_ranges.update(yr_matches[0])
    elif len(yr_matches) > 1:
        for i, match in enumerate(yr_matches, 1):
            key = f"<YEAR_RANGE_{i}>"
            value = normalize_dash("–".join(match))
            c1[key] = value.strip()
            used_years_in_ranges.update(match)

    # LINKS
    links = iter_matches_in_order(text, c1_patterns["LINK"])
    c1.update(assign_numbered("LINK", links))

    # YEARS (excluding those used in ranges)
    all_years = iter_matches_in_order(text, c1_patterns["YEAR"])
    years = [y for y in all_years if y not in used_years_in_ranges]
    c1.update(assign_numbered("YEAR", years))

    # DOCUMENT_TYPE
    doc_types = iter_matches_in_order(text, c1_patterns["DOCUMENT_TYPE"])
    c1.update(assign_numbered("DOCUMENT_TYPE", doc_types))

    # DATES — collect occurrences with spans to classify context
    date_occurrences = []  # list of (raw_date, start, end, context_label or None)
    date_union = re.compile(
        f"(?:{c1_patterns['DATE_ISO'].pattern})|"
        f"(?:{c1_patterns['DATE_TEXTUAL'].pattern})|"
        f"(?:{c1_patterns['DATE_EU'].pattern})",
        re.I
    )
    for m in date_union.finditer(text):
        raw = m.group(0)
        start, end = m.span()
        # Look around +/- 60 chars for context words
        left = max(0, start - 60)
        right = min(len(text), end + 60)
        ctx = classify_text_date_context(text[left:right])
        date_occurrences.append((raw, start, end, ctx))

    # Assign special placeholders first
    used_placeholders = set()
    for ph_name in ("Last_Reviewed_Date", "Deployment_Date", "Last_Executed_Date"):
        for i, (raw, _, _, ctx) in enumerate(date_occurrences):
            if ctx == ph_name and raw not in date_assignments:
                date_assignments[raw] = f"<{ph_name}>"
                used_placeholders.add(ph_name)

    # Assign generic DATE_1 and DATE_2 to remaining dates in first-appearance order
    generic_dates = [raw for (raw, _, _, ctx) in date_occurrences if raw not in date_assignments]
    if generic_dates:
        date_assignments[generic_dates[0]] = "<DATE_1>"
    if len(generic_dates) > 1:
        date_assignments[generic_dates[1]] = "<DATE_2>"

    # Keep raw date values in C1 ONLY for discoverability (not used directly in transform)
    # (You may remove this if you don't want DATE keys at all in c1)
    if generic_dates:
        c1.update(assign_numbered("DATE", generic_dates[:2]))

    # ---------- C2 Extraction from text ----------
    for label, pattern in partial_patterns.items():
        values = iter_matches_in_order(text, pattern)
        c2.update(assign_numbered(label, values))

    # ---------- From files (collect date neighbor meta; not used for replacing unless present in text) ----------
    def process_df(df, filename):
        # Ensure string dtype for scanning
        df = df.astype(str)

        date_cell_patterns = [
            c1_patterns["DATE_ISO"],
            c1_patterns["DATE_TEXTUAL"],
            c1_patterns["DATE_EU"],
        ]

        # For later: add non-date label hits discovered in files into c2
        file_hits_accumulator = defaultdict(list)

        for _, row in df.iterrows():
            row_matches = []  # ordered list of dicts: {label, value, column}

            for col in df.columns:
                cell = row[col].strip()
                if not cell or cell.lower() in ("nan", "none"):
                    continue

                # dates
                for dp in date_cell_patterns:
                    for m in dp.finditer(cell):
                        row_matches.append({"label": "DATE", "value": m.group(0), "column": col})

                # non-date entities (one per label per cell)
                for label, pat in partial_patterns.items():
                    if label == "DATE":
                        continue
                    m = pat.search(cell)
                    if m:
                        row_matches.append({"label": label, "value": m.group(0), "column": col})

            if not row_matches:
                continue

            # de-dup by (label,value,column)
            seen = set()
            ordered = []
            for itm in row_matches:
                key = (itm["label"], itm["value"], itm["column"])
                if key not in seen:
                    seen.add(key)
                    ordered.append(itm)

            # date neighbor meta + classification
            for idx, itm in enumerate(ordered):
                if itm["label"] != "DATE":
                    continue
                # neighbor: prefer previous non-date else next non-date
                neighbor = None
                for j in range(idx - 1, -1, -1):
                    if ordered[j]["label"] != "DATE":
                        neighbor = ordered[j]
                        break
                if neighbor is None:
                    for j in range(idx + 1, len(ordered)):
                        if ordered[j]["label"] != "DATE":
                            neighbor = ordered[j]
                            break

                date_raw = itm["value"]
                meta_entry = {
                    "date_raw": date_raw,
                    "date_iso": to_iso_date(date_raw),
                    "source_file": filename,
                    "date_column": itm["column"],
                    "neighbor_label": neighbor["label"] if neighbor else None,
                    "neighbor_value": neighbor["value"] if neighbor else None,
                    "neighbor_column": neighbor["column"] if neighbor else None,
                }
                meta["dates"].append(meta_entry)

                # If the exact raw date string appears in the TEXT, we may also assign a special placeholder
                # according to neighbor context (but we won't force insert if it's not in the text).
                ph = classify_file_date_context(
                    meta_entry["neighbor_label"], meta_entry["neighbor_column"], meta_entry["neighbor_value"]
                )
                if ph and (ph in DATE_ALLOWED_PLACEHOLDERS) and (date_raw in text) and (date_raw not in date_assignments):
                    date_assignments[date_raw] = f"<{ph}>"

            # accumulate other hits to c2 later
            for itm in ordered:
                if itm["label"] == "DATE":
                    continue
                file_hits_accumulator[itm["label"]].append(itm["value"])

        # append discovered non-date hits to c2
        for label, values in file_hits_accumulator.items():
            uniq = []
            seenv = set()
            for v in values:
                if v not in seenv:
                    seenv.add(v)
                    uniq.append(v)
            if uniq:
                add_values_from_files(label, uniq, c2)

    if os.path.isdir(data_dir):
        for filename in os.listdir(data_dir):
            if "_c2_" in filename and filename.lower().endswith((".xlsx", ".xls", ".csv")):
                filepath = os.path.join(data_dir, filename)
                try:
                    if filename.lower().endswith(".csv"):
                        df = pd.read_csv(filepath, dtype=str, encoding="utf-8", errors="ignore")
                        process_df(df, filename)
                    else:
                        xls = pd.ExcelFile(filepath)
                        for sheet in xls.sheet_names:
                            df = xls.parse(sheet, dtype=str)
                            process_df(df, f"{filename}::{sheet}")
                except Exception as e:
                    print(f"Error processing {filename}: {e}")

    return {"c1": c1, "c2": c2, "meta": {"dates": meta["dates"], "date_assignments": date_assignments}}


# ----------------------------
# Transformation (dates restricted to the 5 placeholders)
# ----------------------------

def _label_of_placeholder(ph: str) -> str:
    """Extract LABEL from a placeholder like <LABEL> or <LABEL_3>.
       Allow letters, digits, underscores, and hyphens.
    """
    m = re.match(r'^<([A-Za-z0-9_-]+)(?:_\d+)?>$', ph)
    return m.group(1) if m else ""


def transform_text(original_text, c1, c2, date_assignments):
    """
    Replace phrases with keys, **but for dates only use**:
      <DATE_1>, <DATE_2>, <Last_Reviewed_Date>, <Deployment_Date>, <Last_Executed_Date>
    """
    phrase_to_key = {}

    # Non-date entities as usual
    for key, value in {**c1, **c2}.items():
        if not value:
            continue
        if isinstance(value, str) and value.lower() == "version":
            continue
        lbl = _label_of_placeholder(key)
        if lbl.startswith("DATE"):
            continue  # we control dates via date_assignments only
        phrase_to_key.setdefault(value, key)

    # Dates: only the assigned ones
    for raw_date, ph in date_assignments.items():
        # guard: only these five placeholders
        lbl = _label_of_placeholder(ph)
        if lbl in DATE_ALLOWED_PLACEHOLDERS:
            phrase_to_key.setdefault(raw_date, ph)

    # Longest-first replacement
    sorted_phrases = sorted(phrase_to_key.items(), key=lambda x: len(x[0]), reverse=True)
    transformed_text = original_text
    for phrase, key in sorted_phrases:
        escaped = re.escape(phrase)
        transformed_text = re.sub(rf'(?<!\w){escaped}(?!\w)', key, transformed_text)

    return transformed_text


# ----------------------------
# CLI
# ----------------------------

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    text_filepath = os.path.join(base_dir, "..", "test.txt")
    data_dir = os.path.join(base_dir, "..", "data", "classification")

    result = extract_c1_and_c2(text_filepath, data_dir)
    print("C1 and C2 Extraction:")
    print({k: v for k, v in result.items() if k != "meta"})  # keep meta separate
    print("Date meta:", result["meta"]["dates"])
    print("Date assignments:", result["meta"]["date_assignments"])

    with open(text_filepath, 'r', encoding='utf-8') as f:
        original_text = f.read()

    transformed_text = transform_text(
        original_text,
        result["c1"],
        result["c2"],
        result["meta"]["date_assignments"]
    )
    print("\nTransformed Text:")
    print(transformed_text)
