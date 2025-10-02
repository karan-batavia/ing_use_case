import re
from typing import List, Dict, Any, Optional
from presidio_analyzer import (
    AnalyzerEngine,
    RecognizerRegistry,
    Pattern,
    PatternRecognizer,
)
from presidio_anonymizer import AnonymizerEngine, OperatorConfig
from presidio_analyzer.nlp_engine import NlpEngineProvider
from prompt_scrubber import PromptScrubber


class BelgianSpecificRecognizer(PatternRecognizer):
    """Custom recognizer for Belgian-specific patterns that require regex"""

    def __init__(self):
        patterns = [
            # Belgian IBAN - very specific format
            Pattern("Belgian IBAN", r"\bBE\d{2}\s?\d{4}\s?\d{4}\s?\d{4}\b", 0.95),
            
            # Belgian National ID - very specific format (multiple variants)
            Pattern("Belgian National ID", r"\b\d{2}\.\d{2}\.\d{2}-\d{3}[\.-]\d{2}\b", 0.95),
            
            # Belgian postal codes
            Pattern("Belgian Postal Code", r"\b[1-9]\d{3}\b", 0.7),
            
            # Belgian phone numbers
            Pattern("Belgian Phone", r"\+32\s?\d{1,2}\s?\d{3}\s?\d{3}\s?\d{3}", 0.9),
            Pattern("Belgian Phone Alt", r"\+32\s?\d{8,9}", 0.85),
            
            # .be email domains (Belgian specific)
            Pattern("Belgian Email", r"\b[\w\.-]+@[\w\.-]+\.be\b", 0.9),
        ]
        super().__init__(
            supported_entity="BELGIAN_SPECIFIC",
            patterns=patterns,
            name="belgian_specific_recognizer",
        )


class FinancialPatternRecognizer(PatternRecognizer):
    """Recognizer for financial patterns that are hard to detect with NLP"""

    def __init__(self):
        patterns = [
            # Product codes - very specific to ING
            Pattern("Product Code", r"\bBP[A-Z0-9]{6,8}\b", 0.9),
            Pattern("ING Product", r"\bING[A-Z]{2}\d+\b", 0.9),
            
            # Corporate keys
            Pattern("Corporate Key", r"\b[A-Z]{2}\d{2}[A-Z]{2}\b", 0.8),
            
            # PIN codes (4 digits in financial context)
            Pattern("PIN Code", r"\bPIN\s+\d{4}\b", 0.9),
            Pattern("PIN Number", r"\busing PIN\s+\d{4}\b", 0.9),
            
            # CCV codes
            Pattern("CCV Code", r"\bCCV\s+\d{3,4}\b", 0.9),
            Pattern("CVV Code", r"\bCVV\s+\d{3,4}\b", 0.9),
            
            # Credit scores in context
            Pattern("Credit Score", r"\bcredit score\s+(?:is\s+)?\d{3}\b", 0.9),
            Pattern("Score Context", r"\bscore\s+(?:of\s+)?\d{3}\b", 0.8),
        ]
        super().__init__(
            supported_entity="FINANCIAL_CODE",
            patterns=patterns,
            name="financial_pattern_recognizer",
        )


class MonetaryAmountRecognizer(PatternRecognizer):
    """Enhanced recognizer for monetary amounts with context"""

    def __init__(self):
        patterns = [
            # European currency formats
            Pattern("Euro Amount", r"€\s?\d{1,3}(?:[,\.]\d{3})*(?:[,\.]\d{2})?", 0.9),
            Pattern("EUR Amount", r"\d{1,3}(?:[,\.]\d{3})*(?:[,\.]\d{2})?\s?EUR", 0.9),
            
            # Context-based monetary amounts
            Pattern("Balance Amount", r"\bbalance\s+of\s+€?\d{1,3}(?:[,\.]\d{3})*(?:[,\.]\d{2})?", 0.95),
            Pattern("Income Amount", r"\bincome\s+of\s+€?\d{1,3}(?:[,\.]\d{3})*(?:[,\.]\d{2})?", 0.95),
            Pattern("Loan Amount", r"\bloan\s+of\s+€?\d{1,3}(?:[,\.]\d{3})*(?:[,\.]\d{2})?", 0.95),
            Pattern("Capital Amount", r"\bcapital\s+€?\d{1,3}(?:[,\.]\d{3})*(?:[,\.]\d{2})?", 0.95),
            Pattern("Transfer Amount", r"\bfor\s+€?\d{1,3}(?:[,\.]\d{3})*(?:[,\.]\d{2})?", 0.9),
            
            # Plain large numbers that are likely monetary
            Pattern("Large Amount", r"\b\d{1,3}(?:,\d{3})+(?:\.\d{2})?\b", 0.7),
        ]
        super().__init__(
            supported_entity="MONETARY_AMOUNT",
            patterns=patterns,
            name="monetary_amount_recognizer",
        )


class AddressRecognizer(PatternRecognizer):
    """Enhanced address recognizer for Belgian/European addresses"""

    def __init__(self):
        patterns = [
            # Belgian address format: postal code + city, street number + street name
            Pattern("Belgian Address", r"\b\d{4}\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*,\s+\d+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*", 0.9),
            
            # Street addresses
            Pattern("Street Address", r"\b\d+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln)", 0.85),
            
            # European style addresses
            Pattern("Euro Address", r"\b\d+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+", 0.7),
        ]
        super().__init__(
            supported_entity="ADDRESS",
            patterns=patterns,
            name="address_recognizer",
        )


class CreditScoreRecognizer(PatternRecognizer):
    """Recognizer for credit scores"""

    def __init__(self):
        patterns = [
            # Credit scores typically 300-850
            Pattern("Credit Score Context", r"\bcredit\s+score\s+(?:is\s+)?([3-8]\d{2})\b", 0.9),
            Pattern("Score Context", r"\bscore\s+(?:is\s+)?([3-8]\d{2})\b", 0.8),
            Pattern("Rating Context", r"\brating\s+(?:is\s+)?([3-8]\d{2})\b", 0.8),
        ]
        super().__init__(
            supported_entity="CREDIT_SCORE",
            patterns=patterns,
            name="credit_score_recognizer",
        )


class UUIDRecognizer(PatternRecognizer):
    """Recognizer for UUID/GUID patterns commonly used in banking transactions"""

    def __init__(self):
        patterns = [
            # Standard UUID format (8-4-4-4-12 hex digits)
            Pattern("UUID v4", r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b", 0.95),
            
            # UUID without hyphens
            Pattern("UUID Compact", r"\b[0-9a-fA-F]{32}\b", 0.8),
            
            # Transaction reference patterns with context
            Pattern("Reference ID", r"\breference\s+ID\s+[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b", 0.98),
            Pattern("Transfer ID", r"\btransfer\s+(?:with\s+)?(?:reference\s+)?ID\s+[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b", 0.98),
        ]
        super().__init__(
            supported_entity="UUID",
            patterns=patterns,
            name="uuid_recognizer",
        )


class SensitiveAttributeRecognizer(PatternRecognizer):
    """Recognizer for sensitive personal attributes like religion, sexuality, ethnicity, political views"""

    def __init__(self):
        patterns = [
            # Religious beliefs - based on your data
            Pattern("Religion Christian", r"\b(?:[Cc]hristian|[Cc]atholiC|[Pp]rotestant|[Oo]rthodox|[Bb]aptist|[Mm]ethodist|[Ll]utheran)\b", 0.9),
            Pattern("Religion Muslim", r"\b(?:[Mm]uslim|[Ii]slam|[Ss]unni|[Ss]hia|[Ii]slamic)\b", 0.9),
            Pattern("Religion Jewish", r"\b(?:[Jj]ewish|[Jj]udaism|[Hh]ebrew|[Oo]rthodox [Jj]ew)\b", 0.9),
            Pattern("Religion Hindu", r"\b(?:[Hh]indu|[Hh]induism|[Bb]uddhist|[Bb]uddhism)\b", 0.9),
            Pattern("Religion Other", r"\b(?:[Aa]theist|[Aa]gnostic|[Nn]one|[Oo]ther)\b", 0.7),
            
            # Sexual orientation - based on your data
            Pattern("Sexual Orientation", r"\b(?:[Hh]eterosexual|[Hh]omosexual|[Bb]isexual|[Aa]sexual|[Gg]ay|[Ll]esbian|[Qq]ueer|LGBTQ)\b", 0.9),
            
            # Ethnicity/Race - based on your data  
            Pattern("Ethnicity", r"\b(?:[Bb]lack|[Ww]hite|[Aa]sian|[Aa]rab|[Mm]ixed|[Hh]ispanic|[Ll]atino|[Cc]aucasian|[Aa]frican)\b", 0.8),
            
            # Political opinions - based on your data
            Pattern("Political Views", r"\b(?:[Ll]iberal|[Cc]onservative|[Cc]entre|[Ll]eft|[Rr]ight|[Dd]emocrat|[Rr]epublican|[Ss]ocialist|[Cc]ommunist)\b", 0.8),
            
            # Health conditions - sensitive medical info
            Pattern("Health Condition", r"\b(?:[Cc]hronic illness|[Mm]ental health|[Dd]iabetes|[Cc]ancer|[Hh]eart disease|[Dd]epression|[Aa]nxiety)\b", 0.9),
            Pattern("Health General", r"\b(?:[Gg]ood|[Pp]oor|[Ee]xcellent)\s+health\b", 0.8),
            
            # Trade union membership
            Pattern("Union Membership", r"\b(?:[Tt]rade [Uu]nion|[Uu]nion [Mm]ember|[Uu]nion [Mm]embership)\b", 0.9),
            Pattern("Union Status", r"\b(?:[Yy]es|[Nn]o)\s+(?:to\s+)?(?:union|trade union)\b", 0.8),
            
            # Criminal history
            Pattern("Criminal Record", r"\b(?:[Cc]riminal [Cc]onvictions?|[Mm]inor|[Mm]ajor|[Nn]one|[Nn]o [Cc]onvictions?)\b", 0.9),
            
            # Philosophical beliefs
            Pattern("Philosophy", r"\b(?:[Aa]theist|[Aa]gnostic|[Hh]umanist|[Ee]xistentialist|[Rr]ationalist)\b", 0.8),
        ]
        super().__init__(
            supported_entity="SENSITIVE_ATTRIBUTE",
            patterns=patterns,
            name="sensitive_attribute_recognizer",
        )


class PresidioNLPBalancedAnonymizer:
    """Hybrid anonymizer that uses PromptScrubber first, then NLP-powered Presidio for comprehensive coverage"""

    def __init__(self):
        # Initialize the domain-specific prompt scrubber first
        self.prompt_scrubber = PromptScrubber()
        print("✅ Initialized PromptScrubber for ING-specific patterns")
        
        # Initialize NLP engine with better model
        nlp_configuration = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}],
        }

        try:
            provider = NlpEngineProvider(nlp_configuration=nlp_configuration)
            nlp_engine = provider.create_engine()
            print("✅ Loaded large spaCy model for better NLP")
        except Exception as e:
            print(f"⚠️ Could not load large spaCy model, using small model: {e}")
            nlp_configuration = {
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
            }
            provider = NlpEngineProvider(nlp_configuration=nlp_configuration)
            nlp_engine = provider.create_engine()

        # Create registry and load built-in recognizers first
        # These use NLP and ML for: PERSON, EMAIL_ADDRESS, PHONE_NUMBER, CREDIT_CARD, etc.
        registry = RecognizerRegistry()
        registry.load_predefined_recognizers(nlp_engine=nlp_engine)
        
        print("✅ Loaded built-in NLP recognizers for common entities")

        # Add our specific regex-based recognizers for Belgian/ING patterns
        registry.add_recognizer(BelgianSpecificRecognizer())
        registry.add_recognizer(FinancialPatternRecognizer())
        registry.add_recognizer(MonetaryAmountRecognizer())
        registry.add_recognizer(AddressRecognizer())
        registry.add_recognizer(CreditScoreRecognizer())
        registry.add_recognizer(UUIDRecognizer())
        registry.add_recognizer(SensitiveAttributeRecognizer())
        
        print("✅ Added custom recognizers for Belgian/financial patterns")

        # Initialize engines
        self.analyzer = AnalyzerEngine(registry=registry, nlp_engine=nlp_engine)
        self.anonymizer = AnonymizerEngine()

        # Configure anonymization operators with better labels
        self.operators_config = {
            # Built-in NLP-detected entities
            "PERSON": OperatorConfig("replace", {"new_value": "[PERSON]"}),
            "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "[EMAIL]"}),
            "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "[PHONE]"}),
            "CREDIT_CARD": OperatorConfig("replace", {"new_value": "[CREDIT_CARD]"}),
            "DATE_TIME": OperatorConfig("replace", {"new_value": "[DATE]"}),
            "IP_ADDRESS": OperatorConfig("replace", {"new_value": "[IP_ADDRESS]"}),
            "URL": OperatorConfig("replace", {"new_value": "[URL]"}),
            "IBAN_CODE": OperatorConfig("replace", {"new_value": "[IBAN]"}),
            "US_SSN": OperatorConfig("replace", {"new_value": "[SSN]"}),
            "LOCATION": OperatorConfig("replace", {"new_value": "[LOCATION]"}),
            
            # Our custom entities
            "BELGIAN_SPECIFIC": OperatorConfig("replace", {"new_value": "[BELGIAN_DATA]"}),
            "FINANCIAL_CODE": OperatorConfig("replace", {"new_value": "[SENSITIVE_CODE]"}),
            "MONETARY_AMOUNT": OperatorConfig("replace", {"new_value": "[AMOUNT]"}),
            "ADDRESS": OperatorConfig("replace", {"new_value": "[ADDRESS]"}),
            "CREDIT_SCORE": OperatorConfig("replace", {"new_value": "[CREDIT_SCORE]"}),
            "UUID": OperatorConfig("replace", {"new_value": "[REFERENCE_ID]"}),
            "TRANSACTION_ID": OperatorConfig("replace", {"new_value": "[REFERENCE_ID]"}),
            "TRANSFER_ID": OperatorConfig("replace", {"new_value": "[REFERENCE_ID]"}),
            "REFERENCE_ID": OperatorConfig("replace", {"new_value": "[REFERENCE_ID]"}),
            "GUID": OperatorConfig("replace", {"new_value": "[REFERENCE_ID]"}),
            "ID": OperatorConfig("replace", {"new_value": "[ID]"}),
            "PIN": OperatorConfig("replace", {"new_value": "[PIN]"}),
            "CCV": OperatorConfig("replace", {"new_value": "[CCV]"}),
            "CVV": OperatorConfig("replace", {"new_value": "[CVV]"}),
            "CODE": OperatorConfig("replace", {"new_value": "[CODE]"}),
            "SENSITIVE_CODE": OperatorConfig("replace", {"new_value": "[CODE]"}),
            "SENSITIVE_ATTRIBUTE": OperatorConfig("replace", {"new_value": "[SENSITIVE_DATA]"}),
            "PRODUCT_CODE": OperatorConfig("replace", {"new_value": "[PRODUCT_CODE]"}),
            "CORPORATE_KEY": OperatorConfig("replace", {"new_value": "[CORP_KEY]"}),
            "AMOUNT": OperatorConfig("replace", {"new_value": "[AMOUNT]"}),
            "BALANCE": OperatorConfig("replace", {"new_value": "[AMOUNT]"}),
            "INCOME": OperatorConfig("replace", {"new_value": "[AMOUNT]"}),
            "LOAN": OperatorConfig("replace", {"new_value": "[AMOUNT]"}),
            "CAPITAL": OperatorConfig("replace", {"new_value": "[AMOUNT]"}),
            "TRANSFER": OperatorConfig("replace", {"new_value": "[AMOUNT]"}),
            "PAYMENT": OperatorConfig("replace", {"new_value": "[AMOUNT]"}),
            "VALUE": OperatorConfig("replace", {"new_value": "[AMOUNT]"}),
            "PRICE": OperatorConfig("replace", {"new_value": "[AMOUNT]"}),
            "COST": OperatorConfig("replace", {"new_value": "[AMOUNT]"}),
            "FEE": OperatorConfig("replace", {"new_value": "[AMOUNT]"}),
            "RATE": OperatorConfig("replace", {"new_value": "[RATE]"})
        }

    def analyze_text(self, text: str) -> List:
        return self.analyzer.analyze(
            text=text,
            language="en",
            entities=None  # means "use everything in the registry"
        )

    def anonymize_text(self, text: str) -> str:
        """
        Hybrid anonymization using both PromptScrubber and Presidio NLP
        
        Step 1: Use Presidio to detect UUIDs and other critical patterns first
        Step 2: Use PromptScrubber for ING-specific exact matches and domain patterns  
        Step 3: Use Presidio NLP again for remaining PII detection with context understanding
        """
        print(f"🔍 Starting hybrid anonymization...")
        
        # Step 1: First pass with Presidio to catch UUIDs and critical patterns
        print("🎯 Step 1: Presidio first pass for UUIDs and critical patterns...")
        critical_entities = ["UUID", "CREDIT_CARD", "IP_ADDRESS", "CREDIT_SCORE"]
        critical_results = self.analyzer.analyze(
            text=text,
            language="en", 
            entities=critical_entities
        )
        
        if critical_results:
            print(f"✅ Found {len(critical_results)} critical entities")
            critical_anonymized = self.anonymizer.anonymize(
                text=text,
                analyzer_results=critical_results,
                operators=self.operators_config
            )
            step1_text = critical_anonymized.text
        else:
            print("ℹ️ No critical entities found in first pass")
            step1_text = text
        
        # Step 2: Apply PromptScrubber for domain-specific patterns
        print("📋 Step 2: Applying PromptScrubber for ING-specific patterns...")
        scrubbed_text = self.prompt_scrubber.scrub_prompt(step1_text)
        
        # Check what was changed in step 2
        changes_made = step1_text != scrubbed_text
        if changes_made:
            print(f"✅ PromptScrubber found and masked domain-specific data")
        else:
            print("ℹ️ No domain-specific patterns found by PromptScrubber")
        
        # Step 3: Apply Presidio NLP on the result to catch remaining PII
        print("🧠 Step 3: Applying Presidio NLP for remaining PII detection...")
        analysis_results = self.analyze_text(scrubbed_text)
        
        if analysis_results:
            print(f"🎯 Presidio found {len(analysis_results)} additional PII entities")
            final_result = self.anonymizer.anonymize(
                text=scrubbed_text,
                analyzer_results=analysis_results,
                operators=self.operators_config
            )
            final_text = final_result.text
        else:
            print("ℹ️ No additional PII found by Presidio")
            final_text = scrubbed_text
        
        print(f"✅ Hybrid anonymization complete!")
        return final_text

    def get_detailed_analysis(self, text: str) -> Dict[str, Any]:
        """Get detailed analysis showing what each step detected"""
        # Step 1: PromptScrubber analysis
        scrubbed_text = self.prompt_scrubber.scrub_prompt(text)
        prompt_scrubber_matches = self.prompt_scrubber.scrub(text)
        
        # Step 2: Presidio analysis on scrubbed text
        presidio_results = self.analyze_text(scrubbed_text)
        
        analysis = {
            "original_text_length": len(text),
            "prompt_scrubber": {
                "matches_found": len(prompt_scrubber_matches) if prompt_scrubber_matches else 0,
                "matches": prompt_scrubber_matches or {},
                "text_after_scrubbing": scrubbed_text
            },
            "presidio_nlp": {
                "entities_found": len(presidio_results),
                "entities": []
            },
            "final_result": self.anonymize_text(text)
        }
        
        # Add Presidio details
        for result in presidio_results:
            detected_text = scrubbed_text[result.start:result.end]
            analysis["presidio_nlp"]["entities"].append({
                "type": result.entity_type,
                "text": detected_text,
                "confidence": result.score,
                "start": result.start,
                "end": result.end
            })
        
        return analysis

    def get_analysis_details(self, text: str) -> Dict[str, Any]:
        """Get detailed analysis of what was detected"""
        results = self.analyze_text(text)
        
        details = {
            "total_entities": len(results),
            "entities_by_type": {},
            "detected_items": []
        }
        
        for result in results:
            entity_type = result.entity_type
            detected_text = text[result.start:result.end]
            
            if entity_type not in details["entities_by_type"]:
                details["entities_by_type"][entity_type] = 0
            details["entities_by_type"][entity_type] += 1
            
            details["detected_items"].append({
                "type": entity_type,
                "text": detected_text,
                "start": result.start,
                "end": result.end,
                "confidence": result.score
            })
        
        # Sort by position in text
        details["detected_items"].sort(key=lambda x: x["start"])
        
        return details


# Test function
def test_balanced_anonymizer():
    """Test the balanced NLP + regex anonymizer"""
    print("🔧 Initializing Presidio NLP Balanced Anonymizer...")
    anonymizer = PresidioNLPBalancedAnonymizer()
    
    test_text = """
Hello, my name is Jane Smith, I'm muslim and my social security is 901231-123.45.
Contact me at jane.smith@example.com or +32 498 12 34 56.
Wire transfer IBAN BE13 0018 6454 3175 and amount EUR 12,500.00 on 2025-08-10.
National number 90.10.10-612-39; FaceID enabled."""
    
    print("\n" + "=" * 50)
    print("ORIGINAL TEXT")
    print("=" * 50)
    print(test_text)
    
    print("\n" + "=" * 50)
    print("HYBRID ANALYSIS DETAILS")
    print("=" * 50)
    details = anonymizer.get_detailed_analysis(test_text)
    
    print(f"� PromptScrubber matches: {details['prompt_scrubber']['matches_found']}")
    if details['prompt_scrubber']['matches']:
        for filename, matches in details['prompt_scrubber']['matches'].items():
            print(f"  � {filename}: {len(matches)} matches")
    
    print(f"🧠 Presidio NLP entities: {details['presidio_nlp']['entities_found']}")
    if details['presidio_nlp']['entities']:
        for entity in details['presidio_nlp']['entities']:
            print(f"  • {entity['type']}: '{entity['text']}' (confidence: {entity['confidence']:.2f})")
    
    print("\n" + "=" * 50)
    print("FINAL ANONYMIZED RESULT")
    print("=" * 50)
    print(details['final_result'])


if __name__ == "__main__":
    test_balanced_anonymizer()