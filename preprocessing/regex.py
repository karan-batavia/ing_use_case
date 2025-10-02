import os
import re
import pandas as pd


def normalize_column(col):
    clean = str(col).strip().replace(";", "_").replace(" ", "_")
    return f"<{clean}>"


def extract_c1_and_c2(text_filepath, data_dir):
    with open(text_filepath, 'r', encoding='utf-8') as f:
        text = f.read()

    text_lower = text.lower()
    all_words = set(re.findall(r'\b\w+\b', text_lower))

    # ---------- C1 Extraction ----------
    c1 = {}
    c1_matched_words = set()
    used_years_in_ranges = set()

    c1_patterns = {
        "link": r'https?://[^\s<>"]+|www\.[^\s<>"]+',
        "year_range": r'\b(19[0-9]{2}|20[0-4][0-9]|2050)\s*[-–]\s*(19[0-9]{2}|20[0-4][0-9]|2050)\b',
        "year": r'\b(19[0-9]{2}|20[0-4][0-9]|2050)\b',
        "document_type": r'\bpillar 3(?:\s+\w+){2,3}|\b(?:\w+\s+){1,2}results\b|\b\w+\s+report\b'
    }

    # First pass: year ranges
    range_matches = re.findall(c1_patterns["year_range"], text)
    for i, match in enumerate(range_matches, 1):
        key = f"<YEAR_RANGE_{i}>"
        value = "–".join(match)
        c1[key] = value.strip()
        used_years_in_ranges.update(match)
        for part in match:
            c1_matched_words.update(re.findall(r'\b\w+\b', part.lower()))

    # Other C1 patterns
    for label, pattern in c1_patterns.items():
        if label == "year_range":
            continue

        matches = re.findall(pattern, text)
        count = 1
        for match in matches:
            if label == "year" and match in used_years_in_ranges:
                continue
            key = f"<{label.upper()}_{count}>"
            value = match if isinstance(match, str) else "–".join(match)
            c1[key] = value.strip()
            count += 1

            if isinstance(match, str):
                c1_matched_words.update(re.findall(r'\b\w+\b', match.lower()))
            else:
                for part in match:
                    c1_matched_words.update(re.findall(r'\b\w+\b', part.lower()))

    # ---------- C2 Extraction ----------
    c2 = {}

    partial_patterns = {
        "VM_Name": r'VM-IAAS-\d+',
        "Network_interface_ID": r'NIC-IAAS-\d+',
        "System_name": r'SYS-IAAS-\d+',
        "Design_Document_Reference": r'\bDOC\b',
        "Fees": r'annual fee',
        "Version": r'v\d+\.\d+',
        "e-mail": r'\b[\w\.-]+@[\w\.-]+\.\w+\b',
        "Phone_Number": r'\+32\s?\d{3}\s?\d{3}\s?\d{3}'
    }

    # Scan the text file itself
    for label, pattern in partial_patterns.items():
        matches = re.findall(pattern, text, re.IGNORECASE)
        for i, match in enumerate(matches, 1):
            key = f"<{label}_{i}>"
            c2[key] = match

    # Process Excel/CSV in data_dir if any
    def process_df(df, global_matched_values):
        for col in df.columns:
            col_key_base = normalize_column(col)
            matches = []
            for i, cell in df[col].dropna().items():
                cell_str = str(cell).strip()
                for label, pattern in partial_patterns.items():
                    partial_match = re.search(pattern, cell_str, re.IGNORECASE)
                    if partial_match:
                        matches.append(partial_match.group(0))
            unique_matches = list(dict.fromkeys(matches))
            for j, match in enumerate(unique_matches, 1):
                c2[f"{col_key_base}_{j}"] = match

    global_matched_values = set()
    for filename in os.listdir(data_dir):
        if "_c2_" in filename and filename.endswith((".xlsx", ".xls", ".csv")):
            filepath = os.path.join(data_dir, filename)
            try:
                if filename.endswith(".csv"):
                    df = pd.read_csv(filepath, dtype=str)
                    process_df(df, global_matched_values)
                else:
                    xls = pd.ExcelFile(filepath)
                    for sheet in xls.sheet_names:
                        df = xls.parse(sheet, dtype=str)
                        process_df(df, global_matched_values)
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    return {"c1": c1, "c2": c2}


def transform_text(original_text, c1, c2):
    """
    Replace matched phrases with their semantic keys from C1 and C2.
    Longest matches replaced first to avoid overlap.
    """
    phrase_to_key = {}
    for key, value in {**c1, **c2}.items():
        if value:
            phrase_to_key[value] = key

    # Sort longest first
    sorted_phrases = sorted(phrase_to_key.items(), key=lambda x: len(x[0]), reverse=True)

    transformed_text = original_text
    for phrase, key in sorted_phrases:
        escaped_phrase = re.escape(phrase)
        transformed_text = re.sub(rf'(?<!\w){escaped_phrase}(?!\w)', key, transformed_text)

    return transformed_text


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    text_filepath = os.path.join(base_dir, "..", "test.txt")
    data_dir = os.path.join(base_dir, "..", "data", "classification")

    result = extract_c1_and_c2(text_filepath, data_dir)
    print("C1 and C2 Extraction:")
    print(result)

    with open(text_filepath, 'r', encoding='utf-8') as f:
        original_text = f.read()

    transformed_text = transform_text(original_text, result["c1"], result["c2"])
    print("\nTransformed Text:")
    print(transformed_text)
