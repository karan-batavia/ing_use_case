"""
This script regroups C4 and C3 regex queries that are going to be used to identify 
what needs to be anonymized in the prompts as well as in the files.

C4:
    - IBAN/Credit card numbers (pin, ccv, expiry date)
    - Transaction numbers (communications)
    - Authentication info (passwords, biometrics, PIN)
    - Social security
    - Phone numbers
    
C3: 
    - Customer name
    - Customer number
    - Email
    - ID info
    - Date of birth
    - 'Born in'
    - Address
    - Citizenship
    - Any employee info like contracts, etc.
"""

import re

# ==============================================================================
# C4 - CRITICAL RISK LEVEL PATTERNS
# ==============================================================================

# 1. IBAN - International Bank Account Numbers
IBAN_REGEX = re.compile(
    r"\b[A-Z]{2}\d{2}[A-Z0-9]{1,30}\b"
)

# 2. Credit Card Numbers
CREDIT_CARD_REGEX = re.compile(
    r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3(?:0[0-5]|[68][0-9])[0-9]{11}|6(?:011|5[0-9]{2})[0-9]{12}|(?:2131|1800|35\d{3})\d{11})\b"
)

# 3. Belgian Social Security Numbers (11 digits: YYMMDD-XXX.XX)
SOCIAL_SECURITY_REGEX = re.compile(
    r"\b\d{2}\.?\d{2}\.?\d{2}[-.\s]?\d{3}[-.\s]?\d{2}\b"
)

# 4. PIN Numbers
PIN_REGEX = re.compile(
    r"\b(?:PIN|pin|Pin|code|CODE)[\s:]*\d{4,6}\b",
    re.IGNORECASE
)

# 5. CVV/CVC Numbers
CVV_REGEX = re.compile(
    r"\b(?:CVV|CVC|code|security|Security Code)[\s:]*\d{3,4}\b",
    re.IGNORECASE
)

# 6. Transaction Numbers/Reference Numbers
TRANSACTION_REGEX = re.compile(
    r"\b(?:TXN|REF|TRANS|Transaction|Reference)[-:\s]*[A-Z0-9]{6,20}\b",
    re.IGNORECASE
)

# 7. Phone Numbers (Belgian & international formats)
PHONE_REGEX = re.compile(
    r"\b(?:\+32|0032|0)[1-9](?:[-.\s]?\d{2,3}){2,3}[-.\s]?\d{2,4}\b|\b(?:\+\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b"
)


# ==============================================================================
# C3 - HIGH RISK LEVEL PATTERNS
# ==============================================================================

# 8. Emails
EMAIL_REGEX = re.compile(
    r"\b[A-Za-z0-9][A-Za-z0-9._%+-]*@[A-Za-z0-9][A-Za-z0-9.-]*\.[A-Za-z]{2,}\b"
)

# 9. Customer Numbers
CUSTOMER_NUMBER_REGEX = re.compile(
    r"\b(?:CUST|CL|CLIENT|ID|Customer|Client)[-:\s]*[A-Z0-9]{4,15}\b",
    re.IGNORECASE
)

# 10. Dates of Birth (various formats)
DATE_OF_BIRTH_REGEX = re.compile(
    r"\b(?:DOB|born|birth|Date of Birth)[\s:]*(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{2,4}[/-]\d{1,2}[/-]\d{1,2})\b",
    re.IGNORECASE
)

# 11. Belgian ID Numbers
BELGIAN_ID_REGEX = re.compile(
    r"\b\d{2}\.?\d{2}\.?\d{2}[-.\s]?\d{3}[-.\s]?\d{2}\b"
)

# 12. Addresses
ADDRESS_REGEX = re.compile(
    r"\b\d{1,5}\s+(?:[A-Z][a-z]+\s?)+(?:straat|str\.?|laan|weg|plein|square|avenue|av\.?|rue|street|st\.?|road|rd\.?|boulevard|blvd\.?|lane|ln\.?|drive|dr\.?|way|court|ct\.?)\b",
    re.IGNORECASE
)

# 13. Names (full names with titles)
NAME_REGEX = re.compile(
    r"\b(?:Mr\.?|Mrs\.?|Ms\.?|Dr\.?|Prof\.?)\s+[A-Z][a-z]+(?:[-'\s][A-Z][a-z]+)+\b"
)


# 14. Postal Codes (Belgian format: 4 digits) - IMPROVED to avoid year confusion
POSTAL_CODE_REGEX = re.compile(
    r"\b(?:postal code|postcode|zip|code postal|PC)[\s:]+([1-9]\d{3})(?:\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)?\b|"
    r"\b([1-9]\d{3})\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b(?!\s*(?:report|disclosure|Q[1-4]|quarter|year))|"
    r"\b([1-9]\d{3})\s+(?:Belgium|BE|Belgique|België)\b",
    re.IGNORECASE
)

# 15. Citizenship/Nationality
CITIZENSHIP_REGEX = re.compile(
    r"\b(?:nationality|citizenship|citizen of|national of)[\s:]+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b",
    re.IGNORECASE
)

# 16. Employee ID Numbers
EMPLOYEE_ID_REGEX = re.compile(
    r"\b(?:EMP|EMPL|Employee ID|Staff ID)[-:\s]*[A-Z0-9]{4,12}\b",
    re.IGNORECASE
)

# 17. Contract Numbers
CONTRACT_NUMBER_REGEX = re.compile(
    r"\b(?:Contract|Agreement|CNT)[-:\s]*[A-Z0-9]{4,15}\b",
    re.IGNORECASE
)


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def is_likely_year(text: str, match_start: int, match_end: int) -> bool:
    """
    Check if a 4-digit match is likely a year rather than a postal code.
    Looks at context before and after the match.
    """
    # Extract context
    context_before = text[max(0, match_start-20):match_start].lower()
    context_after = text[match_end:min(len(text), match_end+20)].lower()
    
    # Year indicators
    year_indicators = ['year', 'in', 'q1', 'q2', 'q3', 'q4', 'quarter', 
                       'report', 'disclosure', 'fiscal', 'fy', 'cy', '–q', '-q']
    
    # Check for year context
    for indicator in year_indicators:
        if indicator in context_before or indicator in context_after:
            return True
    
    # Check if it's in a range that's more likely a year (1900-2099)
    match_text = text[match_start:match_end]
    if match_text.isdigit():
        num = int(match_text)
        if 1900 <= num <= 2099:
            # If followed by specific year indicators, it's definitely a year
            if any(ind in context_after[:15] for ind in ['report', 'disclosure', 'pillar', 'q1', 'q2', 'q3', 'q4']):
                return True
    
    return False


def filter_postal_codes(text: str, matches: list) -> list:
    """
    Filter out false positive postal code matches (like years).
    """
    filtered = []
    for match in matches:
        start, end = match.span()
        if not is_likely_year(text, start, end):
            filtered.append(match)
    return filtered


# ==============================================================================
# PATTERN DICTIONARY
# ==============================================================================

ALL_PATTERNS = {
    # C4 - Critical Risk
    'iban': IBAN_REGEX,
    'credit_card': CREDIT_CARD_REGEX,
    'social_security': SOCIAL_SECURITY_REGEX,
    'pin': PIN_REGEX,
    'cvv': CVV_REGEX,
    'transaction': TRANSACTION_REGEX,
    'phone': PHONE_REGEX,
    
    # C3 - High Risk
    'email': EMAIL_REGEX,
    'customer_number': CUSTOMER_NUMBER_REGEX,
    'date_of_birth': DATE_OF_BIRTH_REGEX,
    'belgian_id': BELGIAN_ID_REGEX,
    'address': ADDRESS_REGEX,
    'name': NAME_REGEX,
    'postal_code': POSTAL_CODE_REGEX,
    'citizenship': CITIZENSHIP_REGEX,
    'employee_id': EMPLOYEE_ID_REGEX,
    'contract_number': CONTRACT_NUMBER_REGEX,
}


# ==============================================================================
# TEST EXAMPLES
# ==============================================================================

if __name__ == "__main__":
    test_cases = [
        # Should NOT match years
        "Create an archive index for 2021 Pillar 3 Disclosures Q1–Q4",
        "The 2020 report shows improvements",
        "In 1995, the company was founded",
        
        # SHOULD match postal codes
        "Send it to 2000 Antwerp, Belgium",
        "Address: Kerkstraat 15, 9000 Gent",
        "Postal code: 1000 Brussels",
        "ZIP: 3000",
        "Lives in 8000 Brugge",
        
        # Edge cases
        "Contact: John Doe, john@example.com, +32 2 123 4567",
        "IBAN: BE68539007547034",
        "Client ID: CUST-12345, born 15/03/1985",
    ]
    
    print("=== TESTING IMPROVED POSTAL CODE DETECTION ===\n")
    
    for i, test_text in enumerate(test_cases, 1):
        print(f"Test {i}: {test_text}")
        
        # Test postal code detection
        postal_matches = list(POSTAL_CODE_REGEX.finditer(test_text))
        if postal_matches:
            filtered = filter_postal_codes(test_text, postal_matches)
            print(f"  Postal codes found: {[m.group() for m in filtered]}")
        else:
            print("  Postal codes found: None")
        
        # Test other patterns
        print(f"  Emails: {EMAIL_REGEX.findall(test_text)}")
        print(f"  Names: {[m.group() for m in NAME_REGEX.finditer(test_text)]}")
        print()