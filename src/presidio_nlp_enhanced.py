import re
from typing import List, Dict, Any, Optional, Union
from presidio_analyzer import (
    AnalyzerEngine,
    RecognizerRegistry,
    Pattern,
    PatternRecognizer,
    EntityRecognizer,
    RecognizerResult,
)
from presidio_anonymizer import AnonymizerEngine, OperatorConfig
from presidio_analyzer.nlp_engine import NlpEngineProvider
import uuid
import random


class BelgianSpecificRecognizer(PatternRecognizer):
    """Custom recognizer for Belgian-specific patterns that require regex"""

    def __init__(self):
        patterns = [
            # Belgian IBAN - very specific format
            Pattern("Belgian IBAN", r"\bBE\d{2}\s?\d{4}\s?\d{4}\s?\d{4}\b", 0.95),
            
            # Belgian National ID - very specific format
            Pattern("Belgian National ID", r"\b\d{2}\.\d{2}\.\d{2}-\d{3}\.\d{2}\b", 0.95),
            
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
        ]
        super().__init__(
            supported_entity="FINANCIAL_CODE",
            patterns=patterns,
            name="financial_pattern_recognizer",
        )


class MonetaryAmountRecognizer(EntityRecognizer):
    """NLP-enhanced recognizer for monetary amounts with context understanding"""

    def __init__(self):
        super().__init__(
            supported_entities=["MONETARY_AMOUNT"],
            name="monetary_amount_recognizer",
        )

    def load(self) -> None:
        pass

    def analyze(self, text: str, entities: List[str], nlp_artifacts=None) -> List[RecognizerResult]:
        """Use NLP to find monetary amounts with context"""
        results = []
        
        if nlp_artifacts is None:
            return results
            
        # Access the spaCy doc from nlp_artifacts safely
        try:
            doc = getattr(nlp_artifacts, 'nlp_doc', None)
            if doc is None:
                # Fallback to pattern-only detection
                doc = None
        except AttributeError:
            doc = None
        
        # Pattern-based detection for explicit monetary amounts
        money_patterns = [
            r"€\s?\d{1,3}(?:,\d{3})*(?:\.\d{2})?",  # €1,234.56
            r"\d{1,3}(?:,\d{3})*(?:\.\d{2})?\s?EUR",  # 1,234.56 EUR
            r"balance\s+of\s+€?\d{1,3}(?:,\d{3})*(?:\.\d{2})?",  # balance of €1,234
            r"income\s+of\s+€?\d{1,3}(?:,\d{3})*(?:\.\d{2})?",  # income of €1,234
            r"loan\s+of\s+€?\d{1,3}(?:,\d{3})*(?:\.\d{2})?",  # loan of €1,234
            r"capital\s+€?\d{1,3}(?:,\d{3})*(?:\.\d{2})?",  # capital €1,234
        ]
        
        for pattern in money_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                results.append(
                    RecognizerResult(
                        entity_type="MONETARY_AMOUNT",
                        start=match.start(),
                        end=match.end(),
                        score=0.9,
                    )
                )
        
        # Use spaCy's MONEY entity detection if available
        if doc is not None:
            for ent in doc.ents:
                if ent.label_ == "MONEY":
                    results.append(
                        RecognizerResult(
                            entity_type="MONETARY_AMOUNT",
                            start=ent.start_char,
                            end=ent.end_char,
                            score=0.8,
                        )
                    )
        
        return results


class ContextualPersonRecognizer(EntityRecognizer):
    """Enhanced person name recognizer using NLP context"""

    def __init__(self):
        super().__init__(
            supported_entities=["PERSON"],
            name="contextual_person_recognizer",
        )

    def load(self) -> None:
        pass

    def analyze(self, text: str, entities: List[str], nlp_artifacts=None) -> List[RecognizerResult]:
        """Use NLP to find person names with better context understanding"""
        results = []
        
        if nlp_artifacts is None:
            return results
            
        doc = nlp_artifacts.nlp_doc
        
        # Use spaCy's PERSON entity detection
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                results.append(
                    RecognizerResult(
                        entity_type="PERSON",
                        start=ent.start_char,
                        end=ent.end_char,
                        score=0.9,
                    )
                )
        
        # Look for contextual patterns that indicate names
        name_contexts = [
            r"my name is\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
            r"partner\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
            r"contact.*?([A-Z][a-z]+\s+[A-Z][a-z]+)",
        ]
        
        for pattern in name_contexts:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                name_start = match.start(1)
                name_end = match.end(1)
                results.append(
                    RecognizerResult(
                        entity_type="PERSON",
                        start=name_start,
                        end=name_end,
                        score=0.85,
                    )
                )
        
        return results


class CreditScoreRecognizer(EntityRecognizer):
    """NLP-enhanced recognizer for credit scores with context"""

    def __init__(self):
        super().__init__(
            supported_entities=["CREDIT_SCORE"],
            name="credit_score_recognizer",
        )

    def load(self) -> None:
        pass

    def analyze(self, text: str, entities: List[str], nlp_artifacts=None) -> List[RecognizerResult]:
        """Find credit scores using context clues"""
        results = []
        
        # Look for credit score patterns with context
        credit_patterns = [
            r"credit score\s+(?:is\s+)?(\d{3})",
            r"score\s+(?:of\s+)?(\d{3})",
            r"credit rating\s+(\d{3})",
        ]
        
        for pattern in credit_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                score_value = int(match.group(1))
                # Credit scores are typically 300-850
                if 300 <= score_value <= 850:
                    results.append(
                        RecognizerResult(
                            entity_type="CREDIT_SCORE",
                            start=match.start(1),
                            end=match.end(1),
                            score=0.9,
                        )
                    )
        
        return results


class AddressRecognizer(EntityRecognizer):
    """NLP-enhanced address recognizer"""

    def __init__(self):
        super().__init__(
            supported_entities=["ADDRESS"],
            name="address_recognizer",
        )

    def load(self) -> None:
        pass

    def analyze(self, text: str, entities: List[str], nlp_artifacts=None) -> List[RecognizerResult]:
        """Find addresses using NLP and patterns"""
        results = []
        
        if nlp_artifacts is None:
            return results
            
        doc = nlp_artifacts.nlp_doc
        
        # Use spaCy's GPE (geopolitical entities) and LOC (locations)
        for ent in doc.ents:
            if ent.label_ in ["GPE", "LOC", "FAC"]:
                results.append(
                    RecognizerResult(
                        entity_type="ADDRESS",
                        start=ent.start_char,
                        end=ent.end_char,
                        score=0.7,
                    )
                )
        
        # Pattern for full addresses
        address_patterns = [
            r"\d{4}\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*,\s+\d+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*",
            r"\d+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln)",
        ]
        
        for pattern in address_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                results.append(
                    RecognizerResult(
                        entity_type="ADDRESS",
                        start=match.start(),
                        end=match.end(),
                        score=0.85,
                    )
                )
        
        return results


class PresidioNLPEnhancedAnonymizer:
    """Enhanced anonymizer that balances NLP and regex approaches"""

    def __init__(self):
        # Initialize NLP engine with larger model for better NER
        nlp_configuration = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}],
        }

        try:
            provider = NlpEngineProvider(nlp_configuration=nlp_configuration)
            nlp_engine = provider.create_engine()
        except Exception as e:
            print(f"Warning: Could not load large spaCy model. Falling back to small model: {e}")
            nlp_configuration = {
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
            }
            provider = NlpEngineProvider(nlp_configuration=nlp_configuration)
            nlp_engine = provider.create_engine()

        # Create registry and load built-in recognizers (these use NLP)
        registry = RecognizerRegistry()
        registry.load_predefined_recognizers(nlp_engine=nlp_engine)

        # Add our custom recognizers
        # Only use regex for very specific patterns
        registry.add_recognizer(BelgianSpecificRecognizer())
        registry.add_recognizer(FinancialPatternRecognizer())
        
        # Use NLP-enhanced recognizers for better context understanding
        registry.add_recognizer(MonetaryAmountRecognizer())
        registry.add_recognizer(ContextualPersonRecognizer())
        registry.add_recognizer(CreditScoreRecognizer())
        registry.add_recognizer(AddressRecognizer())

        # Initialize engines
        self.analyzer = AnalyzerEngine(registry=registry, nlp_engine=nlp_engine)
        self.anonymizer = AnonymizerEngine()

        # Configure anonymization operators
        self.operators_config = {
            "PERSON": OperatorConfig("replace", {"new_value": "[PERSON]"}),
            "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "[EMAIL]"}),
            "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "[PHONE]"}),
            "CREDIT_CARD": OperatorConfig("replace", {"new_value": "[CREDIT_CARD]"}),
            "DATE_TIME": OperatorConfig("replace", {"new_value": "[DATE]"}),
            "IP_ADDRESS": OperatorConfig("replace", {"new_value": "[IP_ADDRESS]"}),
            "URL": OperatorConfig("replace", {"new_value": "[URL]"}),
            "IBAN_CODE": OperatorConfig("replace", {"new_value": "[IBAN]"}),
            
            # Our custom entities
            "BELGIAN_SPECIFIC": OperatorConfig("replace", {"new_value": "[BELGIAN_DATA]"}),
            "FINANCIAL_CODE": OperatorConfig("replace", {"new_value": "[FINANCIAL_CODE]"}),
            "MONETARY_AMOUNT": OperatorConfig("replace", {"new_value": "[AMOUNT]"}),
            "CREDIT_SCORE": OperatorConfig("replace", {"new_value": "[CREDIT_SCORE]"}),
            "ADDRESS": OperatorConfig("replace", {"new_value": "[ADDRESS]"}),
        }

    def analyze_text(self, text: str) -> List[RecognizerResult]:
        """Analyze text to find PII entities using NLP + custom patterns"""
        return self.analyzer.analyze(
            text=text,
            language="en",
            entities=[
                "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD", 
                "DATE_TIME", "IP_ADDRESS", "URL", "IBAN_CODE",
                "BELGIAN_SPECIFIC", "FINANCIAL_CODE", "MONETARY_AMOUNT",
                "CREDIT_SCORE", "ADDRESS"
            ]
        )

    def anonymize_text(self, text: str) -> str:
        """Anonymize text using both NLP and pattern-based detection"""
        # First, use Presidio's analysis
        analysis_results = self.analyze_text(text)
        
        # Anonymize using Presidio
        anonymized_result = self.anonymizer.anonymize(
            text=text,
            analyzer_results=analysis_results,
            operators=self.operators_config
        )
        
        return anonymized_result.text

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
        
        return details


# Test function
def test_nlp_enhanced_anonymizer():
    """Test the NLP-enhanced anonymizer"""
    anonymizer = PresidioNLPEnhancedAnonymizer()
    
    test_text = """
    Hi there, I need help with my ING account setup. My name is Sarah De Wit and I live at 2000 Brussels, 425 Avenue Louise. You can reach me at +32 47 892 3456 or sarah.dewit@telenet.be. 

    My national ID is 88.05.24-234.87 and I'm employed at Accenture Belgium with an annual income of €78,450. I have a savings account BE74 3201 2345 6789 with a current balance of €45,230.15.

    I'm interested in the INGCC5 credit card product (product code: BPMSVFCA) and would like to apply for a personal loan of €25,000. My credit score is 689 and I have no previous loan defaults.

    I recently made a transfer with reference ID f8a92b45-cc31-4e7b-9f12-d4c589e2a7b3 to my spouse's account BE95 4049 5490 6589 for €2,500 on 2025-09-15. The transaction was processed through our mobile banking app using PIN 7428.
    """
    
    print("=" * 50)
    print("ORIGINAL TEXT")
    print("=" * 50)
    print(test_text)
    
    print("\n" + "=" * 50)
    print("ANALYSIS DETAILS")
    print("=" * 50)
    details = anonymizer.get_analysis_details(test_text)
    print(f"Total entities detected: {details['total_entities']}")
    print(f"Entities by type: {details['entities_by_type']}")
    
    print("\n" + "=" * 50)
    print("ANONYMIZED RESULT")
    print("=" * 50)
    anonymized = anonymizer.anonymize_text(test_text)
    print(anonymized)


if __name__ == "__main__":
    test_nlp_enhanced_anonymizer()