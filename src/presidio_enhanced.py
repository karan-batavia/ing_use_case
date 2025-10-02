import re
from typing import List, Dict, Any, Optional, Union
from presidio_analyzer import (
    AnalyzerEngine,
    RecognizerRegistry,
    Pattern,
    PatternRecognizer,
)
from presidio_anonymizer import AnonymizerEngine, OperatorConfig
from presidio_analyzer.nlp_engine import NlpEngineProvider
import uuid
import random


class BelgianIBANRecognizer(PatternRecognizer):
    """Custom recognizer for Belgian IBAN numbers"""

    def __init__(self):
        patterns = [
            Pattern("Belgian IBAN", r"\bBE\d{2}\s?\d{4}\s?\d{4}\s?\d{4}\b", 0.9),
            Pattern("IBAN Format", r"\bBE\d{14}\b", 0.9),
        ]
        super().__init__(
            supported_entity="BELGIAN_IBAN",
            patterns=patterns,
            name="belgian_iban_recognizer",
        )


class BelgianNationalIDRecognizer(PatternRecognizer):
    """Custom recognizer for Belgian National ID numbers"""

    def __init__(self):
        patterns = [
            Pattern(
                "Belgian National ID", r"\b\d{2}\.\d{2}\.\d{2}-\d{3}\.\d{2}\b", 0.9
            ),
        ]
        super().__init__(
            supported_entity="BELGIAN_NATIONAL_ID",
            patterns=patterns,
            name="belgian_national_id_recognizer",
        )


class ProductCodeRecognizer(PatternRecognizer):
    """Custom recognizer for ING product codes"""

    def __init__(self):
        patterns = [
            Pattern("ING Product Code", r"\bING[A-Z]{1,4}\d*\b", 0.8),
            Pattern("Generic Product Code", r"\bBP[A-Z0-9]{6,8}\b", 0.8),
            Pattern("Application Code", r"\b[A-Z]{2,4}_\d+\b", 0.7),
        ]
        super().__init__(
            supported_entity="PRODUCT_CODE",
            patterns=patterns,
            name="product_code_recognizer",
        )


class DocumentReferenceRecognizer(PatternRecognizer):
    """Custom recognizer for document references and IDs"""

    def __init__(self):
        patterns = [
            Pattern("Document Reference", r"\bDOC[A-Z0-9]{6,8}\b", 0.9),
            Pattern("VM Reference", r"\bVM-[A-Z]{2,6}-\d+\b", 0.8),
            Pattern("Network Interface", r"\bNIC-[A-Z]{2,6}-\d+\b", 0.8),
            Pattern("System Reference", r"\bSYS-[A-Z]{2,6}-\d+\b", 0.8),
        ]
        super().__init__(
            supported_entity="DOCUMENT_REFERENCE",
            patterns=patterns,
            name="document_reference_recognizer",
        )


class CorporateKeyRecognizer(PatternRecognizer):
    """Custom recognizer for corporate keys and employee IDs"""

    def __init__(self):
        patterns = [
            Pattern("Corporate Key", r"\b[A-Z]{2}\d{2}[A-Z]{2}\b", 0.8),
            Pattern("Config Group", r"\bT\d{5}\b", 0.7),
        ]
        super().__init__(
            supported_entity="CORPORATE_KEY",
            patterns=patterns,
            name="corporate_key_recognizer",
        )


class BelgianPhoneRecognizer(PatternRecognizer):
    """Custom recognizer for Belgian phone numbers"""

    def __init__(self):
        patterns = [
            Pattern("Belgian Phone", r"\+32\s?\d{2}\s?\d{3}\s?\d{3}", 0.9),
        ]
        super().__init__(
            supported_entity="BELGIAN_PHONE",
            patterns=patterns,
            name="belgian_phone_recognizer",
        )


class BelgianEmailRecognizer(PatternRecognizer):
    """Custom recognizer for Belgian email patterns"""

    def __init__(self):
        patterns = [
            Pattern("ING Email", r"\b[a-zA-Z0-9._%+-]+@ing\.com\b", 0.9),
            Pattern(
                "Belgian Email",
                r"\b[a-zA-Z0-9._%+-]+@(skynet|proximus|telenet)\.be\b",
                0.8,
            ),
        ]
        super().__init__(
            supported_entity="BELGIAN_EMAIL",
            patterns=patterns,
            name="belgian_email_recognizer",
        )


class UUIDRecognizer(PatternRecognizer):
    """Custom recognizer for UUIDs and transaction IDs"""

    def __init__(self):
        patterns = [
            Pattern(
                "UUID",
                r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b",
                0.9,
            ),
            Pattern("Customer ID", r"\bCUST-\d{4}\b", 0.8),
            Pattern("Payment Order", r"\bPO-\d{4}\b", 0.8),
        ]
        super().__init__(
            supported_entity="UUID_IDENTIFIER",
            patterns=patterns,
            name="uuid_recognizer",
        )


class SensitiveAttributeRecognizer(PatternRecognizer):
    """Custom recognizer for sensitive personal attributes"""

    def __init__(self):
        patterns = [
            Pattern(
                "Sexual Orientation",
                r"\b(Heterosexual|Homosexual|Bisexual|Asexual)\b",
                0.8,
            ),
            Pattern(
                "Religious Belief",
                r"\b(Christian|Muslim|Jewish|Hindu|Buddhist|Atheist|Agnostic)\b",
                0.7,
            ),
            Pattern("Health Status", r"\b(Chronic illness|Mental health)\b", 0.8),
            Pattern("Political Opinion", r"\b(Liberal|Conservative|Green)\b", 0.6),
            Pattern("Ethnic Origin", r"\b(Black|White|Asian|Arab|Mixed)\b", 0.7),
        ]
        super().__init__(
            supported_entity="SENSITIVE_ATTRIBUTE",
            patterns=patterns,
            name="sensitive_attribute_recognizer",
        )


class MonetaryAmountRecognizer(PatternRecognizer):
    """Custom recognizer for monetary amounts"""

    def __init__(self):
        patterns = [
            # European format with € symbol
            Pattern("Euro Amount", r"€\s?\d{1,3}(?:,\d{3})*(?:\.\d{2})?", 0.9),
            # Plain amounts that look like money
            Pattern("Large Amount", r"\b\d{1,3}(?:,\d{3})+(?:\.\d{2})?\b", 0.7),
            Pattern("Decimal Amount", r"\b\d+\.\d{2}\b", 0.6),
            # Specific patterns for the test data
            Pattern("Income Pattern", r"\b€\d{2,3},\d{3}\b", 0.8),
            Pattern("Balance Pattern", r"\b€\d{2,3},\d{3}\.\d{2}\b", 0.8),
        ]
        super().__init__(
            supported_entity="MONETARY_AMOUNT",
            patterns=patterns,
            name="monetary_amount_recognizer",
        )


class CreditScoreRecognizer(PatternRecognizer):
    """Custom recognizer for credit scores"""

    def __init__(self):
        patterns = [
            # Credit score in context
            Pattern(
                "Credit Score Context", r"\bcredit\s+score\s+is\s+(\d{3,4})\b", 0.9
            ),
            Pattern("Score Context", r"\bscore\s+(\d{3,4})\b", 0.8),
            # Look for 3-digit numbers in credit context
            Pattern(
                "Credit Score Range",
                r"\b[3-8]\d{2}\b(?=.*(?:credit|score|rating))",
                0.7,
            ),
            # Standalone credit score pattern
            Pattern(
                "Standalone Score", r"\b(?:credit\s+score|score):\s*(\d{3,4})\b", 0.8
            ),
        ]
        super().__init__(
            supported_entity="CREDIT_SCORE",
            patterns=patterns,
            name="credit_score_recognizer",
        )


class PINRecognizer(PatternRecognizer):
    """Custom recognizer for PIN codes"""

    def __init__(self):
        patterns = [
            # More direct pattern for PIN followed by numbers
            Pattern("PIN Context", r"\bPIN\s+(\d{4,6})\b", 0.9),
            Pattern("Banking PIN", r"\busing\s+PIN\s+(\d{4,6})\b", 0.9),
            # Look for 4-digit numbers in banking context
            Pattern(
                "Numeric PIN", r"\b\d{4}\b(?=.*(?:PIN|banking|app|transaction))", 0.7
            ),
            # Standalone 4-digit numbers that could be PINs
            Pattern("Standalone PIN", r"\bPIN\s+\d{4}\b", 0.8),
        ]
        super().__init__(
            supported_entity="PIN_CODE",
            patterns=patterns,
            name="pin_recognizer",
        )


class CCVRecognizer(PatternRecognizer):
    """Custom recognizer for credit card CCV codes"""

    def __init__(self):
        patterns = [
            # CCV in context
            Pattern("CCV Context", r"\bCCV\s+(\d{3,4})\b", 0.9),
            Pattern("CVV Context", r"\bCVV\s+(\d{3,4})\b", 0.9),
            # Look for 3-digit numbers near credit card context
            Pattern(
                "CCV Pattern", r"\b\d{3}\b(?=.*(?:expiring|CCV|CVV|credit|card))", 0.7
            ),
            # Direct CCV pattern
            Pattern("CCV Direct", r"\bCCV\s+\d{3,4}\b", 0.8),
        ]
        super().__init__(
            supported_entity="CCV_CODE",
            patterns=patterns,
            name="ccv_recognizer",
        )
        patterns = [
            Pattern("CCV Code", r"\b(?:CCV|CVV|CVC)\s+(\d{3,4})\b", 0.9),
            Pattern("Security Code", r"\b\d{3,4}\b(?=.*(?:CCV|CVV|CVC|security))", 0.8),
        ]
        super().__init__(
            supported_entity="CCV_CODE",
            patterns=patterns,
            name="ccv_recognizer",
        )


class BelgianPostalCodeRecognizer(PatternRecognizer):
    """Custom recognizer for Belgian postal codes"""

    def __init__(self):
        patterns = [
            Pattern("Belgian Postal Code", r"\b[1-9]\d{3}\b(?=\s+[A-Z][a-z]+)", 0.8),
            Pattern("Postal Code Pattern", r"\b[1-9]\d{3}\s+Brussels\b", 0.9),
        ]
        super().__init__(
            supported_entity="POSTAL_CODE",
            patterns=patterns,
            name="postal_code_recognizer",
        )


class StreetAddressRecognizer(PatternRecognizer):
    """Custom recognizer for street addresses"""

    def __init__(self):
        patterns = [
            Pattern(
                "Street Address",
                r"\b\d+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Street|St|Avenue|Ave|Road|Rd|Lane|Ln|Drive|Dr|Boulevard|Blvd)\b",
                0.8,
            ),
            Pattern(
                "European Address",
                r"\b\d+\s+(?:Avenue|Rue|Straat)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b",
                0.8,
            ),
        ]
        super().__init__(
            supported_entity="STREET_ADDRESS",
            patterns=patterns,
            name="street_address_recognizer",
        )


class PresidioEnhanced:
    """Enhanced Presidio anonymizer for ING banking data with Belgian-specific patterns"""

    def __init__(self):
        # Initialize NLP engine
        nlp_configuration = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}],
        }

        try:
            provider = NlpEngineProvider(nlp_configuration=nlp_configuration)
            nlp_engine = provider.create_engine()
        except Exception as e:
            print(f"Warning: Could not load spaCy model. Using basic NLP engine: {e}")
            # Fallback to basic configuration
            nlp_configuration = {
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
            }
            provider = NlpEngineProvider(nlp_configuration=nlp_configuration)
            nlp_engine = provider.create_engine()

        # Create registry and add custom recognizers
        registry = RecognizerRegistry()
        registry.load_predefined_recognizers(nlp_engine=nlp_engine)

        # Add our custom recognizers
        registry.add_recognizer(BelgianIBANRecognizer())
        registry.add_recognizer(BelgianNationalIDRecognizer())
        registry.add_recognizer(ProductCodeRecognizer())
        registry.add_recognizer(DocumentReferenceRecognizer())
        registry.add_recognizer(CorporateKeyRecognizer())
        registry.add_recognizer(BelgianPhoneRecognizer())
        registry.add_recognizer(BelgianEmailRecognizer())
        registry.add_recognizer(UUIDRecognizer())
        registry.add_recognizer(SensitiveAttributeRecognizer())
        registry.add_recognizer(MonetaryAmountRecognizer())
        registry.add_recognizer(CreditScoreRecognizer())
        registry.add_recognizer(PINRecognizer())
        registry.add_recognizer(CCVRecognizer())
        registry.add_recognizer(BelgianPostalCodeRecognizer())
        registry.add_recognizer(StreetAddressRecognizer())

        # Initialize analyzer and anonymizer
        self.analyzer = AnalyzerEngine(registry=registry, nlp_engine=nlp_engine)
        self.anonymizer = AnonymizerEngine()

        # Load classification data for exact matching
        self.classification_data = self._load_classification_data()

    def _load_classification_data(self) -> Dict[str, List[str]]:
        """Load classification data from files for exact matching"""
        data = {
            "products": ["INGBA1", "INGBA2", "INGCC1", "INGM1", "INGPL1", "INGSA1"],
            "companies": ["ING", "Azure", "AWS", "Google Cloud"],
        }

        return data

    def analyze_text(self, text: str):
        """Analyze text and return detected entities"""
        # Define entities to look for
        entities = [
            "PERSON",
            "EMAIL_ADDRESS",
            "PHONE_NUMBER",
            "CREDIT_CARD",
            "IBAN_CODE",
            "IP_ADDRESS",
            "DATE_TIME",
            # Custom entities
            "BELGIAN_IBAN",
            "BELGIAN_NATIONAL_ID",
            "PRODUCT_CODE",
            "DOCUMENT_REFERENCE",
            "CORPORATE_KEY",
            "BELGIAN_PHONE",
            "BELGIAN_EMAIL",
            "UUID_IDENTIFIER",
            "SENSITIVE_ATTRIBUTE",
        ]

        try:
            results = self.analyzer.analyze(
                text=text, entities=entities, language="en", score_threshold=0.5
            )
            return results
        except Exception as e:
            print(f"Warning: Analysis failed: {e}")
            return []

    def anonymize_text(
        self, text: str, anonymization_config: Optional[Dict] = None
    ) -> str:
        """Anonymize text using custom configuration"""
        # Analyze text first
        analyzer_results = self.analyze_text(text)

        if not analyzer_results:
            # If analysis failed, do basic regex-based anonymization
            return self._basic_anonymization(text)

        # Default anonymization operators
        operators = {
            "PERSON": OperatorConfig("replace", {"new_value": "[PERSON]"}),
            "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "[EMAIL]"}),
            "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "[PHONE]"}),
            "CREDIT_CARD": OperatorConfig("replace", {"new_value": "[CREDIT_CARD]"}),
            "IBAN_CODE": OperatorConfig("replace", {"new_value": "[IBAN]"}),
            "IP_ADDRESS": OperatorConfig("replace", {"new_value": "[IP_ADDRESS]"}),
            "DATE_TIME": OperatorConfig("replace", {"new_value": "[DATE]"}),
            # Custom entities
            "BELGIAN_IBAN": OperatorConfig("replace", {"new_value": "[BELGIAN_IBAN]"}),
            "BELGIAN_NATIONAL_ID": OperatorConfig(
                "replace", {"new_value": "[NATIONAL_ID]"}
            ),
            "PRODUCT_CODE": OperatorConfig("replace", {"new_value": "[PRODUCT_CODE]"}),
            "DOCUMENT_REFERENCE": OperatorConfig("replace", {"new_value": "[DOC_REF]"}),
            "CORPORATE_KEY": OperatorConfig("replace", {"new_value": "[CORP_KEY]"}),
            "BELGIAN_PHONE": OperatorConfig("replace", {"new_value": "[PHONE_BE]"}),
            "BELGIAN_EMAIL": OperatorConfig("replace", {"new_value": "[EMAIL_BE]"}),
            "UUID_IDENTIFIER": OperatorConfig("replace", {"new_value": "[UUID]"}),
            "SENSITIVE_ATTRIBUTE": OperatorConfig(
                "replace", {"new_value": "[SENSITIVE_DATA]"}
            ),
            "MONETARY_AMOUNT": OperatorConfig("replace", {"new_value": "[AMOUNT]"}),
            "CREDIT_SCORE": OperatorConfig("replace", {"new_value": "[CREDIT_SCORE]"}),
            "PIN_CODE": OperatorConfig("replace", {"new_value": "[PIN]"}),
            "CCV_CODE": OperatorConfig("replace", {"new_value": "[CCV]"}),
            "POSTAL_CODE": OperatorConfig("replace", {"new_value": "[POSTAL_CODE]"}),
            "STREET_ADDRESS": OperatorConfig("replace", {"new_value": "[ADDRESS]"}),
        }

        # Override with custom config if provided
        if anonymization_config:
            operators.update(anonymization_config)

        try:
            # Anonymize
            anonymized_result = self.anonymizer.anonymize(
                text=text, analyzer_results=analyzer_results, operators=operators
            )

            # Apply additional exact match anonymization
            final_text = self._apply_exact_match_anonymization(anonymized_result.text)

            return final_text
        except Exception as e:
            print(f"Warning: Anonymization failed: {e}")
            return self._basic_anonymization(text)

    def _basic_anonymization(self, text: str) -> str:
        """Basic regex-based anonymization as fallback"""
        # Belgian IBAN
        text = re.sub(r"\bBE\d{2}\s?\d{4}\s?\d{4}\s?\d{4}\b", "[BELGIAN_IBAN]", text)

        # Belgian National ID
        text = re.sub(r"\b\d{2}\.\d{2}\.\d{2}-\d{3}\.\d{2}\b", "[NATIONAL_ID]", text)

        # Email addresses
        text = re.sub(
            r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b", "[EMAIL]", text
        )

        # Phone numbers
        text = re.sub(r"\+32\s?\d{2}\s?\d{3}\s?\d{3}", "[PHONE]", text)

        # UUIDs
        text = re.sub(
            r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
            "[UUID]",
            text,
        )

        # Monetary amounts - enhanced patterns
        text = re.sub(r"€\s?\d{1,3}(?:,\d{3})*(?:\.\d{2})?", "[AMOUNT]", text)
        text = re.sub(r"\b\d{1,3}(?:,\d{3})+(?:\.\d{2})?\b", "[AMOUNT]", text)

        # Credit scores (3-digit numbers in credit context)
        text = re.sub(
            r"\bcredit\s+score\s+(?:is\s+)?(\d{3,4})\b",
            r"credit score is [CREDIT_SCORE]",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"\b[3-8]\d{2}\b(?=.*(?:credit|score))",
            "[CREDIT_SCORE]",
            text,
            flags=re.IGNORECASE,
        )

        # PIN codes
        text = re.sub(r"\bPIN\s+(\d{4,6})\b", "PIN [PIN]", text, flags=re.IGNORECASE)
        text = re.sub(
            r"\busing\s+PIN\s+(\d{4,6})\b", "using PIN [PIN]", text, flags=re.IGNORECASE
        )

        # CCV codes
        text = re.sub(r"\bCCV\s+(\d{3,4})\b", "CCV [CCV]", text, flags=re.IGNORECASE)
        text = re.sub(r"\bCVV\s+(\d{3,4})\b", "CVV [CCV]", text, flags=re.IGNORECASE)

        # Belgian postal codes
        text = re.sub(
            r"\b[1-9]\d{3}\b(?=.*(?:Brussels|Antwerp|Ghent|Bruges|Leuven|Mons))",
            "[POSTAL_CODE]",
            text,
        )

        # Street addresses (number + street name)
        text = re.sub(
            r"\b\d{1,4}\s+[A-Z][a-zA-Z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Lane|Dr|Drive)\b",
            "[ADDRESS]",
            text,
        )

        # Product codes
        text = re.sub(r"\bING[A-Z]{1,4}\d*\b", "[PRODUCT_CODE]", text)
        text = re.sub(r"\bBP[A-Z0-9]{6,8}\b", "[PRODUCT_CODE]", text)

        # Document references
        text = re.sub(r"\bDOC[A-Z0-9]{6,8}\b", "[DOC_REF]", text)

        # Corporate keys
        text = re.sub(r"\b[A-Z]{2}\d{2}[A-Z]{2}\b", "[CORP_KEY]", text)

        return text

    def _apply_exact_match_anonymization(self, text: str) -> str:
        """Apply exact match anonymization for classification data"""
        for category, items in self.classification_data.items():
            for item in items:
                if item in text:
                    text = text.replace(item, f"[{category.upper()}_NAME]")

        return text

    def get_detection_summary(self, text: str) -> Dict[str, Any]:
        """Get a summary of detected entities"""
        analyzer_results = self.analyze_text(text)

        summary = {
            "total_entities": len(analyzer_results),
            "entity_types": {},
            "high_risk_entities": [],
            "detected_entities": [],
        }

        for result in analyzer_results:
            entity_type = result.entity_type
            if entity_type not in summary["entity_types"]:
                summary["entity_types"][entity_type] = 0
            summary["entity_types"][entity_type] += 1

            summary["detected_entities"].append(
                {
                    "type": entity_type,
                    "text": text[result.start : result.end],
                    "confidence": result.score,
                    "start": result.start,
                    "end": result.end,
                }
            )

            # Mark high-risk entities
            high_risk_types = [
                "BELGIAN_NATIONAL_ID",
                "CREDIT_CARD",
                "SENSITIVE_ATTRIBUTE",
                "BELGIAN_IBAN",
                "UUID_IDENTIFIER",
            ]
            if entity_type in high_risk_types:
                summary["high_risk_entities"].append(
                    {
                        "type": entity_type,
                        "text": text[result.start : result.end],
                        "confidence": result.score,
                    }
                )

        return summary


# Test function
def test_presidio_enhanced():
    """Test the enhanced Presidio anonymizer"""

    print("Initializing Enhanced Presidio Anonymizer...")
    presidio = PresidioEnhanced()

    # Test text with various Belgian data patterns
    test_text1 = """
    Customer Emily Williams (email: emily.williams@hotmail.com, phone: +32 23 361 947) 
    has account BE97649232964609 with national ID 99.11.05-171.40. 
    Transfer ID 65884d64-e8b0-42c5-a7c2-977284b6b0c6 for amount 7192.29 EUR 
    to IBAN BE95 4049 5490 6589. Product code INGBA1 with document reference DOCH92Y1C.
    Corporate key IU19AK for employee Edward Richardson.
    Health status: Chronic illness, Sexual orientation: Heterosexual, Religion: Christian.
    """

    test_text = """
    Hi there, I need help with my ING account setup. My name is Sarah De Wit and I live at 2000 Brussels, 425 Avenue Louise. You can reach me at +32 47 892 3456 or sarah.dewit@telenet.be. 

My national ID is 88.05.24-234.87 and I'm employed at Accenture Belgium with an annual income of €78,450. I have a savings account BE74 3201 2345 6789 with a current balance of €45,230.15.

I'm interested in the INGCC5 credit card product (product code: BPMSVFCA) and would like to apply for a personal loan of €25,000. My credit score is 689 and I have no previous loan defaults.

I recently made a transfer with reference ID f8a92b45-cc31-4e7b-9f12-d4c589e2a7b3 to my spouse's account BE95 4049 5490 6589 for €2,500 on 2025-09-15. The transaction was processed through our mobile banking app using PIN 7428.

My employment details: I work as a Senior Consultant at Accenture, my corporate key is SC47XY, and my work email is s.dewit@accenture.be. I also have a company credit card 5234 1234 5678 9012 with CCV 456 expiring 03/27.

For emergency contacts, please use my partner Jan Peeters (jan.peeters@gmail.com, +32 9 234 5678) who lives at the same address. His national ID is 85.11.02-567.29.

I'm also interested in your investment portfolio services - specifically the ING Sustainable Growth Fund. My risk profile assessment shows I'm a moderate investor with €150,000 in available capital.

Could you help me set up these services and ensure my data complies with GDPR and AML regulations? I understand you'll need to perform KYC verification as part of the onboarding process.

Additional notes: I bank primarily through your mobile app and ATM network. My last login was from IP address 192.168.1.45 on 2025-09-28 at 14:30 CET.
    """

    print("Original text:")
    print(test_text)

    print("\n" + "=" * 50)
    print("DETECTION ANALYSIS")
    print("=" * 50)

    # Get detection summary
    summary = presidio.get_detection_summary(test_text)
    print(f"Total entities detected: {summary['total_entities']}")
    print(f"Entity types: {summary['entity_types']}")
    print(f"High-risk entities: {len(summary['high_risk_entities'])}")

    print("\n" + "=" * 50)
    print("ANONYMIZED RESULT")
    print("=" * 50)

    # Anonymize the text
    anonymized = presidio.anonymize_text(test_text)
    print(anonymized)


if __name__ == "__main__":
    test_presidio_enhanced()
