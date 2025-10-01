import json
import random
import os
from typing import List, Dict, Tuple, Set
from faker import Faker
import spacy
from spacy.tokens import Doc, Span
from regex import extract_c1_and_c2
# Try to load SpaCy model, fall back to smaller model if needed
try:
    nlp = spacy.load("en_core_web_lg")
except OSError:
    print("Warning: en_core_web_lg not found, using en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Import regex patterns
from regex_queries import (
    ALL_PATTERNS, 
    is_likely_year, 
    filter_postal_codes,
    POSTAL_CODE_REGEX
)

# Initialize Faker
fake = Faker()
Faker.seed(42)
random.seed(42)

# ==============================================================================
# SENSITIVITY MAPPING
# ==============================================================================

SENSITIVITY_MAP = {
    # C4 - Critical Risk
    "iban": "C4",
    "credit_card": "C4",
    "social_security": "C4",
    "pin": "C4",
    "cvv": "C4",
    "transaction": "C4",
    "phone": "C4",
    
    # C3 - High Risk
    "email": "C3",
    "customer_number": "C3",
    "date_of_birth": "C3",
    "belgian_id": "C3",
    "address": "C3",
    "name": "C3",
    "postal_code": "C3",
    "citizenship": "C3",
    "employee_id": "C3",
    "contract_number": "C3",
    
    # C2 - Medium Risk (from regex.py)
    "vm_name": "C2",
    "network_interface_id": "C2",
    "system_name": "C2",
    "design_document_reference": "C2",
    "fees": "C2",
    "version": "C2",
    "c2_email": "C2",
    "c2_phone": "C2",
    
    # C1 - Low Risk (from regex.py)
    "link": "C1",
    "year": "C1",
    "year_range": "C1",
    "document_type": "C1",
    
    # SpaCy detected - contextual
    "religion": "C4",  # Special category
    "ethnicity": "C4",  # Special category
    "sexual_orientation": "C4",  # Special category
    "political_opinion": "C4",  # Special category
    "health_condition": "C4",  # Special category
}

CONTEXT_WINDOW = 30

# ==============================================================================
# CONTEXTUAL KEYWORDS FOR SPACY DETECTION
# ==============================================================================

RELIGION_KEYWORDS = {
    'christian', 'christianity', 'muslim', 'islam', 'islamic', 'jewish', 'judaism',
    'hindu', 'hinduism', 'buddhist', 'buddhism', 'catholic', 'protestant', 
    'orthodox', 'sikh', 'sikhism', 'atheist', 'agnostic', 'religious'
}

ETHNICITY_KEYWORDS = {
    'caucasian', 'african', 'asian', 'hispanic', 'latino', 'latina', 'indigenous',
    'native', 'middle eastern', 'pacific islander', 'ethnicity', 'ethnic', 'race',
    'racial', 'ancestry', 'descent', 'heritage'
}

SEXUAL_ORIENTATION_KEYWORDS = {
    'gay', 'lesbian', 'bisexual', 'transgender', 'lgbt', 'lgbtq', 'queer',
    'homosexual', 'heterosexual', 'sexual orientation', 'orientation',
    'same-sex', 'gender identity', 'non-binary', 'cisgender'
}

POLITICAL_KEYWORDS = {
    'socialist', 'conservative', 'liberal', 'democrat', 'republican',
    'communist', 'fascist', 'capitalist', 'marxist', 'libertarian',
    'political party', 'political view', 'political opinion', 'votes for',
    'supports', 'left-wing', 'right-wing'
}

HEALTH_KEYWORDS = {
    'diabetes', 'cancer', 'hiv', 'aids', 'depression', 'anxiety', 'diagnosis',
    'diagnosed with', 'suffers from', 'medical condition', 'illness', 'disease',
    'disorder', 'syndrome', 'mental health', 'physical health', 'treatment for'
}

# Trade union keywords
TRADE_UNION_KEYWORDS = {
    'union member', 'trade union', 'labor union', 'union membership',
    'union representative', 'collective bargaining', 'union dues'
}

# ==============================================================================
# ENTITY INJECTION
# ==============================================================================

def inject_entities(prompt: str) -> Tuple[str, List[Dict]]:
    """
    Inject fake sensitive entities into a raw prompt.
    Return modified text and entity annotations.
    """
    modified_prompt = prompt
    injected_entities = []
    
    # Generate fake data
    name = fake.name()
    iban = fake.iban()
    email = fake.email()
    phone = fake.phone_number()
    address = fake.address().replace("\n", ", ")
    ssn = fake.ssn()
    
    # Inject entities based on context or randomly
    if "client" in prompt.lower() or "customer" in prompt.lower() or random.random() > 0.7:
        modified_prompt += f" Client name: {name}."
        injected_entities.append({
            "entity": name,
            "type": "name",
            "sensitivity": "C3",
            "injected": True
        })
    
    if "transfer" in prompt.lower() or "payment" in prompt.lower() or "account" in prompt.lower():
        modified_prompt += f" IBAN: {iban}."
        injected_entities.append({
            "entity": iban,
            "type": "iban",
            "sensitivity": "C4",
            "injected": True
        })
    
    if "email" in prompt.lower() or "contact" in prompt.lower() or random.random() > 0.6:
        modified_prompt += f" Contact email: {email}."
        injected_entities.append({
            "entity": email,
            "type": "email",
            "sensitivity": "C3",
            "injected": True
        })
    
    if "phone" in prompt.lower() or "call" in prompt.lower() or random.random() > 0.7:
        modified_prompt += f" Phone: {phone}."
        injected_entities.append({
            "entity": phone,
            "type": "phone",
            "sensitivity": "C4",
            "injected": True
        })
    
    if "address" in prompt.lower() or "location" in prompt.lower() or random.random() > 0.8:
        modified_prompt += f" Address: {address}."
        injected_entities.append({
            "entity": address,
            "type": "address",
            "sensitivity": "C3",
            "injected": True
        })
    
    if "social security" in prompt.lower() or "ssn" in prompt.lower() or random.random() > 0.9:
        modified_prompt += f" SSN: {ssn}."
        injected_entities.append({
            "entity": ssn,
            "type": "social_security",
            "sensitivity": "C4",
            "injected": True
        })
    
    return modified_prompt.strip(), injected_entities


# ==============================================================================
# C1/C2 DETECTION (from regex.py)
# ==============================================================================

def detect_c1_c2_entities(text: str, data_dir: str = None) -> List[Dict]:
    """
    Use the extract_c1_and_c2 function from regex.py to detect C1 and C2 entities.
    
    Args:
        text: The text to analyze
        data_dir: Optional directory containing Excel/CSV files for C2 extraction
    
    Returns:
        List of detected C1/C2 entities with metadata
    """
    # Create a temporary file for the text (required by extract_c1_and_c2)
    import tempfile
    detected_entities = []
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as tmp:
        tmp.write(text)
        tmp_path = tmp.name
    
    try:
        # Extract C1 and C2 using regex.py function
        # Use empty data_dir if not provided
        result = extract_c1_and_c2(tmp_path, data_dir or tempfile.gettempdir())
        
        # Process C1 entities
        for key, value in result.get('c1', {}).items():
            # Find the position of this value in the original text
            start = text.find(value)
            if start != -1:
                end = start + len(value)
                
                # Determine entity type from the key
                entity_type = key.split('_')[0].strip('<>').lower()
                
                context_start = max(0, start - CONTEXT_WINDOW)
                context_end = min(len(text), end + CONTEXT_WINDOW)
                
                detected_entities.append({
                    "entity": value,
                    "type": entity_type,
                    "sensitivity": "C1",
                    "span": [start, end],
                    "context": text[context_start:context_end],
                    "detection_method": "c1_regex",
                    "semantic_key": key
                })
        
        # Process C2 entities
        for key, value in result.get('c2', {}).items():
            start = text.find(value)
            if start != -1:
                end = start + len(value)
                
                # Extract entity type from key
                entity_type = key.split('_')[0].strip('<>').lower()
                
                context_start = max(0, start - CONTEXT_WINDOW)
                context_end = min(len(text), end + CONTEXT_WINDOW)
                
                detected_entities.append({
                    "entity": value,
                    "type": entity_type,
                    "sensitivity": "C2",
                    "span": [start, end],
                    "context": text[context_start:context_end],
                    "detection_method": "c2_regex",
                    "semantic_key": key
                })
    
    finally:
        # Clean up temporary file
        os.unlink(tmp_path)
    
    return detected_entities


# ==============================================================================
# REGEX-BASED DETECTION
# ==============================================================================

def detect_entities_with_regex(text: str) -> List[Dict]:
    """
    Use regex patterns to detect and label all entities in the text.
    Returns a list of detected entities with their positions and context.
    """
    detected_entities = []
    
    for entity_type, pattern in ALL_PATTERNS.items():
        matches = list(pattern.finditer(text))
        
        # Special handling for postal codes to filter out years
        if entity_type == 'postal_code':
            matches = filter_postal_codes(text, matches)
        
        for match in matches:
            match_text = match.group()
            start, end = match.span()
            
            # Extract context around the match
            context_start = max(0, start - CONTEXT_WINDOW)
            context_end = min(len(text), end + CONTEXT_WINDOW)
            context = text[context_start:context_end]
            
            detected_entities.append({
                "entity": match_text,
                "type": entity_type,
                "sensitivity": SENSITIVITY_MAP.get(entity_type, "UNKNOWN"),
                "span": [start, end],
                "context": context,
                "detection_method": "regex"
            })
    
    return detected_entities


# ==============================================================================
# SPACY-BASED CONTEXTUAL DETECTION
# ==============================================================================

def detect_contextual_entities(text: str) -> List[Dict]:
    """
    Use SpaCy NER and contextual analysis to detect sensitive categories.
    Focuses on: religion, ethnicity, sexual orientation, political opinions, health info.
    """
    doc = nlp(text)
    contextual_entities = []
    text_lower = text.lower()
    
    # Helper function to check if a keyword is in context
    def find_keyword_context(keywords: Set[str], entity_type: str):
        for keyword in keywords:
            if keyword in text_lower:
                # Find the position of the keyword
                start = text_lower.find(keyword)
                end = start + len(keyword)
                
                # Extract context
                context_start = max(0, start - CONTEXT_WINDOW)
                context_end = min(len(text), end + CONTEXT_WINDOW)
                context = text[context_start:context_end]
                
                contextual_entities.append({
                    "entity": text[start:end],
                    "type": entity_type,
                    "sensitivity": SENSITIVITY_MAP.get(entity_type, "C4"),
                    "span": [start, end],
                    "context": context,
                    "detection_method": "spacy_contextual"
                })
    
    # Detect religion mentions
    find_keyword_context(RELIGION_KEYWORDS, "religion")
    
    # Detect ethnicity mentions
    find_keyword_context(ETHNICITY_KEYWORDS, "ethnicity")
    
    # Detect sexual orientation mentions
    find_keyword_context(SEXUAL_ORIENTATION_KEYWORDS, "sexual_orientation")
    
    # Detect political opinions
    find_keyword_context(POLITICAL_KEYWORDS, "political_opinion")
    
    # Detect health information
    find_keyword_context(HEALTH_KEYWORDS, "health_condition")
    
    # Detect trade union membership
    find_keyword_context(TRADE_UNION_KEYWORDS, "trade_union")
    
    # Also extract PERSON entities from SpaCy that weren't caught by regex
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            # Check if this person entity overlaps with regex-detected names
            # (we'll deduplicate later)
            contextual_entities.append({
                "entity": ent.text,
                "type": "person_spacy",
                "sensitivity": "C3",
                "span": [ent.start_char, ent.end_char],
                "context": text[max(0, ent.start_char - CONTEXT_WINDOW):
                              min(len(text), ent.end_char + CONTEXT_WINDOW)],
                "detection_method": "spacy_ner"
            })
    
    return contextual_entities


# ==============================================================================
# DEDUPLICATION
# ==============================================================================

def deduplicate_entities(entities: List[Dict]) -> List[Dict]:
    """
    Remove duplicate entities based on span overlap.
    Prioritizes regex detection over SpaCy for structured data.
    """
    if not entities:
        return []
    
    # Sort by start position
    sorted_entities = sorted(entities, key=lambda x: x["span"][0])
    
    deduplicated = []
    for entity in sorted_entities:
        # Check if this entity overlaps with any already added
        overlaps = False
        for existing in deduplicated:
            # Check for span overlap
            if not (entity["span"][1] <= existing["span"][0] or 
                   entity["span"][0] >= existing["span"][1]):
                overlaps = True
                # If regex method overlaps with spacy, keep regex
                if entity["detection_method"] == "regex" and existing["detection_method"] != "regex":
                    # Replace existing with regex version
                    deduplicated.remove(existing)
                    deduplicated.append(entity)
                break
        
        if not overlaps:
            deduplicated.append(entity)
    
    # Sort again by position
    return sorted(deduplicated, key=lambda x: x["span"][0])


# ==============================================================================
# COMBINED DETECTION
# ==============================================================================

def label_prompt(prompt: str, inject: bool = False, use_spacy: bool = True, use_c1_c2: bool = True, data_dir: str = None) -> Dict:
    """
    Label a prompt using both regex and SpaCy detection.
    
    Args:
        prompt: The input text to label
        inject: Whether to inject fake entities first
        use_spacy: Whether to use SpaCy for contextual detection
        use_c1_c2: Whether to use C1/C2 extraction from regex.py
        data_dir: Directory for Excel/CSV files (C2 extraction)
    
    Returns:
        Dictionary with labeled text and detected entities
    """
    injected_info = []
    
    # Inject entities if requested
    if inject:
        prompt, injected_info = inject_entities(prompt)
    
    # Detect all entities using regex (C3/C4)
    regex_entities = detect_entities_with_regex(prompt)
    
    # Detect C1/C2 entities
    c1_c2_entities = []
    if use_c1_c2:
        c1_c2_entities = detect_c1_c2_entities(prompt, data_dir)
    
    # Detect contextual entities using SpaCy
    contextual_entities = []
    if use_spacy:
        contextual_entities = detect_contextual_entities(prompt)
    
    # Combine and deduplicate
    all_entities = regex_entities + c1_c2_entities + contextual_entities
    deduplicated_entities = deduplicate_entities(all_entities)
    
    # Calculate statistics
    c4_count = sum(1 for e in deduplicated_entities if e["sensitivity"] == "C4")
    c3_count = sum(1 for e in deduplicated_entities if e["sensitivity"] == "C3")
    c2_count = sum(1 for e in deduplicated_entities if e["sensitivity"] == "C2")
    c1_count = sum(1 for e in deduplicated_entities if e["sensitivity"] == "C1")
    
    regex_count = sum(1 for e in deduplicated_entities if e["detection_method"] == "regex")
    spacy_count = sum(1 for e in deduplicated_entities if e["detection_method"] in ["spacy_contextual", "spacy_ner"])
    c1_c2_count = sum(1 for e in deduplicated_entities if e["detection_method"] in ["c1_regex", "c2_regex"])
    
    return {
        "text": prompt,
        "entities": deduplicated_entities,
        "injected_entities": injected_info if inject else [],
        "statistics": {
            "total_entities": len(deduplicated_entities),
            "c4_count": c4_count,
            "c3_count": c3_count,
            "c2_count": c2_count,
            "c1_count": c1_count,
            "regex_detected": regex_count,
            "spacy_detected": spacy_count,
            "c1_c2_detected": c1_c2_count
        }
    }


# ==============================================================================
# FILE PROCESSING
# ==============================================================================

def load_prompts(file_path: str) -> List[str]:
    """Load prompts from a text file."""
    with open(file_path, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts


def label_all_prompts(input_file: str, output_file: str, inject: bool = False, use_spacy: bool = True, use_c1_c2: bool = True, data_dir: str = None):
    """
    Process all prompts from input file and save labeled data to output file.
    
    Args:
        input_file: Path to input file with raw prompts
        output_file: Path to output JSON file
        inject: Whether to inject fake entities into prompts
        use_spacy: Whether to use SpaCy for contextual detection
        use_c1_c2: Whether to use C1/C2 extraction
        data_dir: Directory for Excel/CSV files (C2 extraction)
    """
    prompts = load_prompts(input_file)
    labeled_data = []
    
    print(f"Processing {len(prompts)} prompts...")
    for i, prompt in enumerate(prompts, 1):
        if i % 10 == 0:
            print(f"  Processed {i}/{len(prompts)} prompts...")
        labeled_data.append(label_prompt(prompt, inject=inject, use_spacy=use_spacy, use_c1_c2=use_c1_c2, data_dir=data_dir))
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save labeled data
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(labeled_data, f, indent=2, ensure_ascii=False)
    
    # Print comprehensive statistics
    total_entities = sum(item["statistics"]["total_entities"] for item in labeled_data)
    total_c4 = sum(item["statistics"]["c4_count"] for item in labeled_data)
    total_c3 = sum(item["statistics"]["c3_count"] for item in labeled_data)
    total_c2 = sum(item["statistics"]["c2_count"] for item in labeled_data)
    total_c1 = sum(item["statistics"]["c1_count"] for item in labeled_data)
    total_regex = sum(item["statistics"]["regex_detected"] for item in labeled_data)
    total_spacy = sum(item["statistics"]["spacy_detected"] for item in labeled_data)
    total_c1_c2 = sum(item["statistics"]["c1_c2_detected"] for item in labeled_data)
    
    print(f"\n{'='*60}")
    print(f"[✓] Labeled {len(labeled_data)} prompts → {output_file}")
    print(f"{'='*60}")
    print(f"Total entities detected: {total_entities}")
    print(f"  - C4 (Critical Risk): {total_c4}")
    print(f"  - C3 (High Risk): {total_c3}")
    print(f"  - C2 (Medium Risk): {total_c2}")
    print(f"  - C1 (Low Risk): {total_c1}")
    print(f"\nDetection methods:")
    print(f"  - Regex patterns (C3/C4): {total_regex}")
    print(f"  - SpaCy contextual: {total_spacy}")
    print(f"  - C1/C2 patterns: {total_c1_c2}")
    
    if inject:
        total_injected = sum(len(item["injected_entities"]) for item in labeled_data)
        print(f"\n[✓] Total entities injected: {total_injected}")
    
    # Entity type breakdown
    entity_type_counts = {}
    for item in labeled_data:
        for entity in item["entities"]:
            entity_type = entity["type"]
            entity_type_counts[entity_type] = entity_type_counts.get(entity_type, 0) + 1
    
    print(f"\nEntity type breakdown:")
    for entity_type, count in sorted(entity_type_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {entity_type}: {count}")


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    INPUT_FILE = "raw_prompts.txt"
    OUTPUT_FILE_NO_INJECT = "labeled_data/labeled_prompts_hybrid.json"
    OUTPUT_FILE_WITH_INJECT = "labeled_data/labeled_prompts_hybrid_with_injection.json"
    OUTPUT_FILE_REGEX_ONLY = "labeled_data/labeled_prompts_regex_only.json"
    
    # Process with hybrid approach (regex + SpaCy), no injection
    print("="*60)
    print("PROCESSING WITH HYBRID APPROACH (Regex + SpaCy)")
    print("="*60)
    label_all_prompts(INPUT_FILE, OUTPUT_FILE_NO_INJECT, inject=False, use_spacy=True)
    
    print("\n\n")
    
    # Process with hybrid approach + injection
    print("="*60)
    print("PROCESSING WITH HYBRID APPROACH + ENTITY INJECTION")
    print("="*60)
    label_all_prompts(INPUT_FILE, OUTPUT_FILE_WITH_INJECT, inject=True, use_spacy=True)
    
    print("\n\n")
    
    # Process with regex only (for comparison)
    print("="*60)
    print("PROCESSING WITH REGEX ONLY (No SpaCy)")
    print("="*60)
    label_all_prompts(INPUT_FILE, OUTPUT_FILE_REGEX_ONLY, inject=False, use_spacy=False)