"""
    This script regroups C4 and C3 regex queries that are going to be used to identify what needs to be anonymized in the prompts as well as in the files.
    C4:
        - IBAN/Credit card numbers (pin, ccv, expiry date)
        - Transaction numbers (communications)
        - Authentification info (passwords, biometrics, PIN)
        - Social security
        - Phone numbers
        
        NOT:
        - Health info
        - Ethnic origin, trade-union membership, criminal convictions/offences/related info, political opinions

    C3: 
        - Customer name
        - Custumer number
        - Email
        - ID info
        - Date of birth
        - 'Born in'
        - Address
        - Citizenship
        - Any employee info like contracts, etc. 

"""
    

import re

# C4 Critical Risk Level Patterns

# 1. IBAN_REGEX
"""Matches International Bank Account Numbers"""
IBAN_REGEX = re.compile(
    r"^AL\d{10}[0-9A-Z]{16}$|^AD\d{10}[0-9A-Z]{12}$|^AT\d{18}$|^BH\d{2}[A-Z]{4}[0-9A-Z]{14}$|^BE\d{14}$|^BA\d{18}$|^BG\d{2}[A-Z]{4}\d{6}[0-9A-Z]{8}$|^HR\d{19}$|^CY\d{10}[0-9A-Z]{16}$|^CZ\d{22}$|^DK\d{16}$|^FO\d{16}$|^GL\d{16}$|^DO\d{2}[0-9A-Z]{4}\d{20}$|^EE\d{18}$|^FI\d{16}$|^FR\d{12}[0-9A-Z]{11}\d{2}$|^GE\d{2}[A-Z]{2}\d{16}$|^DE\d{20}$|^GI\d{2}[A-Z]{4}[0-9A-Z]{15}$|^GR\d{9}[0-9A-Z]{16}$|^HU\d{26}$|^IS\d{24}$|^IE\d{2}[A-Z]{4}\d{14}$|^IL\d{21}$|^IT\d{2}[A-Z]\d{10}[0-9A-Z]{12}$|^[A-Z]{2}\d{5}[0-9A-Z]{13}$|^KW\d{2}[A-Z]{4}22!$|^LV\d{2}[A-Z]{4}[0-9A-Z]{13}$|^LB\d{6}[0-9A-Z]{20}$|^LI\d{7}[0-9A-Z]{12}$|^LT\d{18}$|^LU\d{5}[0-9A-Z]{13}$|^MK\d{5}[0-9A-Z]{10}\d{2}$|^MT\d{2}[A-Z]{4}\d{5}[0-9A-Z]{18}$|^MR13\d{23}$|^MU\d{2}[A-Z]{4}\d{19}[A-Z]{3}$|^MC\d{12}[0-9A-Z]{11}\d{2}$|^ME\d{20}$|^NL\d{2}[A-Z]{4}\d{10}$|^NO\d{13}$|^PL\d{10}[0-9A-Z]{,16}n$|^PT\d{23}$|^RO\d{2}[A-Z]{4}[0-9A-Z]{16}$|^SM\d{2}[A-Z]\d{10}[0-9A-Z]{12}$|^SA\d{4}[0-9A-Z]{18}$|^RS\d{20}$|^SK\d{22}$|^SI\d{17}$|^ES\d{22}$|^SE\d{22}$|^CH\d{7}[0-9A-Z]{12}$|^TN59\d{20}$|^TR\d{7}[0-9A-Z]{17}$|^AE\d{21}$|^GB\d{2}[A-Z]{4}\d{14}$"
)

# 2. Credit Card Numbers
CREDIT_CARD_REGEX = re.compile(
    r"\b(?<!\d.)(3[47]\d{2}([ -]?)(?!(\d)\3{5}|123456|234567|345678|424242|545454)\d{6}\2(?!(\d)\4{4})\d{5}|((4\d|5[1-5]|65)\d{2}|6011)([ -]?)(?!(\d)\8{3}|4242|5454|1234|3456|5678|2345|4567)\d{4}\7(?!(\d)\9{3})\d{4}\7\d{4})(\b|\s)(?!.\d\d)"
)

# 3. Belgian Social Security Numbers (11 digits: YYMMDD-XXX.XX)
SOCIAL_SECURITY_REGEX = re.compile(
    r"\b\d{2}\.?\d{2}\.?\d{2}[-.\s]?\d{3}[-.\s]?\d{2}\b|\b\d{11}\b"
)

# 4. PIN Numbers: Finds PIN codes by looking for the keywords "PIN", "pin", or "code" followed by 4-6 digits, making it context-aware to avoid false positives.
PIN_REGEX = re.compile(
    r"\b(?:PIN|pin|Pin|code)[\s:]*\d{4,6}\b",
    re.IGNORECASE
)

# 5. CVV/CVC Numbers: Identifies CVV/CVC security codes by searching for related keywords ("CVV", "CVC") followed by 3-4 digits.
CVV_REGEX = re.compile(
    r"\b(?:CVV|CVC)[\s:]*\d{3,4}\b",
    re.IGNORECASE
)

# 6. Transaction Numbers/Reference Numbers: Matches transaction reference numbers that start with prefixes like "TXN", "REF", or "TRANS" followed by 6-20 alphanumeric characters.
TRANSACTION_REGEX = re.compile(
    r"\b(?:TXN|REF|TRANS)[-:\s]*[A-Z0-9]{6,20}\b",
    re.IGNORECASE
)

# 7. Phone Numbers (for Belgian formats)
PHONE_REGEX = re.compile(
    r"\b(?:\+\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b"
)


# C3 risk level

# 8. Emails - comprehensive pattern for various email formats
# Matches emails like: user@domain.com, user.name+tag@domain.co.uk, etc.
EMAIL_REGEX = re.compile(
    r"\b[A-Za-z0-9][A-Za-z0-9._%+-]*@[A-Za-z0-9][A-Za-z0-9.-]*\.[A-Za-z]{2,}\b"
)

# 9. Customer Numbers -- Needs to be adapted to ING's format for the exact match (couldn't find an format example online)
CUSTOMER_NUMBER_REGEX = re.compile(
    r"\b(?:CUST|CL|CLIENT|ID)[-:\s]*[A-Z0-9]{4,15}\b|CLIENT\s+\d{6,12}\b",
    re.IGNORECASE
)

# 10. Dates of Birth (various formats)
DATE_OF_BIRTH_REGEX = re.compile(
    r"\b(?:DOB|born|birth)[\s:]*(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{2,4}[/-]\d{1,2}[/-]\d{1,2})\b|"
    r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
    re.IGNORECASE
)

# 11. Belgian ID Numbers
BELGIAN_ID_REGEX = re.compile(
    r"\b\d{2}\.?\d{2}\.?\d{2}[-.\s]?\d{3}[-.\s]?\d{2}\b"
)


# 12. Addresses
ADDRESS_REGEX = re.compile(
    r"\b\d{1,5}\s+(?:[A-Z][a-z]+\s?)+(?:straat|str|laan|weg|plein|square|avenue|av|rue|street|st|road|rd|boulevard|blvd|lane|ln|drive|dr|way|court|ct)\b",
    re.IGNORECASE
)

# 13. Names (improved pattern for full names)
NAME_REGEX = re.compile(
    r"\b(?:Mr\.?|Mrs\.?|Mlle\.?|Mv\.?|Ms\.?|Dr\.?)?\s*([A-Z][a-z]+(?:[-\s][A-Z][a-z]+)*)\s+([A-Z][a-z]+(?:[-\s][A-Z][a-z]+)*)\b"
)

# 14. Postal Codes (Belgian format: 4 digits)
POSTAL_CODE_REGEX = re.compile(
    r"\b(?:(?:postal code|postcode|zip|code postal)[\s:]*)?(?:[1-9]\d{3})\s+(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b|"
    r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*,?\s+([1-9]\d{3})\b|"
    r"\b([1-9]\d{3})\s+(?:Belgium|BE|Belgique|België)\b",
    re.IGNORECASE
)

# 15. Citizenship/Nationality
CITIZENSHIP_REGEX = re.compile(
    r"\b(?:nationality|nationalieteit|nationalite|citizenship|citizen|national)[\s:]*[A-Z][a-z]+\b",
    re.IGNORECASE
)

# Dictionary for easy access to all patterns
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
}

# Example usage
if __name__ == "__main__":
    test_text = """
    Create an archive index for 2021 Pillar 3 Disclosures Q1–Q4: Q1 https://assets.ing.com/m/1b95f821bd2bc0cc/original/2021-ING-Belgium-Pillar-III-Disclosures-1Q.xlsx ; Q2 https://assets.ing.com/m/362f3cfe7ce92619/original/2021-ING-Belgium-Pillar-III-Disclosures-2Q.xlsx ; Q3 https://assets.ing.com/m/1cccdb7e18606501/original/2021-ING-Belgium-Pillar-III-Disclosures-3Q.xlsx ; Q4 https://assets.ing.com/m/3e95b1f2f3d00a3d/original/2021-ING-Belgium-Pillar-III-Disclosures-4Q.xlsx.
    """

    print("=== C4 CRITICAL RISK PATTERNS ===")
    print("IBANs:", IBAN_REGEX.findall(test_text))
    print("Credit Cards:", CREDIT_CARD_REGEX.findall(test_text))
    print("Social Security:", SOCIAL_SECURITY_REGEX.findall(test_text))
    print("PINs:", PIN_REGEX.findall(test_text))
    print("CVVs:", CVV_REGEX.findall(test_text))
    print("Transactions:", TRANSACTION_REGEX.findall(test_text))
    print("Phones:", PHONE_REGEX.findall(test_text))
    
    print("\n=== C3 HIGH RISK PATTERNS ===")
    print("Emails:", EMAIL_REGEX.findall(test_text))
    print("Customer Numbers:", CUSTOMER_NUMBER_REGEX.findall(test_text))
    print("Dates of Birth:", DATE_OF_BIRTH_REGEX.findall(test_text))
    print("Belgian IDs:", BELGIAN_ID_REGEX.findall(test_text))
    print("Addresses:", ADDRESS_REGEX.findall(test_text))
    print("Names:", NAME_REGEX.findall(test_text))
    print("Postal Codes:", POSTAL_CODE_REGEX.findall(test_text))
    print("Citizenship:", CITIZENSHIP_REGEX.findall(test_text))