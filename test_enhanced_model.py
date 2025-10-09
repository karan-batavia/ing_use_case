#!/usr/bin/env python3
"""
Test Script for Enhanced Redaction Model
Run this to test your new redaction model independently
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.enhanced_redaction_model import EnhancedRedactionModel
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_basic_redaction():
    """Test basic redaction functionality"""
    print("🧪 Testing Enhanced Redaction Model")
    print("=" * 50)

    # Initialize model
    try:
        model = EnhancedRedactionModel()
        print("✅ Model initialized successfully")
    except Exception as e:
        print(f"❌ Model initialization failed: {e}")
        return False

    # Test cases
    test_cases = [
        {
            "name": "Banking Email",
            "text": "Dear John Smith, your account NL91ABNA0417164300 has been credited with EUR 5,234.56. Contact us at support@ing.com or +31 20 123 4567.",
            "expected_entities": ["PERSON_NAME", "IBAN", "AMOUNT", "EMAIL", "PHONE"],
        },
        {
            "name": "Employee Data",
            "text": "Edward Richardson, email: edward.richardson@ing.com, phone: +32 37 441 142, CorpKey: OE27VJ",
            "expected_entities": ["PERSON_NAME", "EMAIL", "PHONE_EU", "CORP_KEY"],
        },
        {
            "name": "Transaction Record",
            "text": "Transaction ID: 550e8400-e29b-41d4-a716-446655440000, amount: EUR 1,500.00, date: 2025-01-15",
            "expected_entities": ["TRANSACTION_ID", "AMOUNT", "DATE"],
        },
    ]

    success_count = 0

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🔍 Test {i}: {test_case['name']}")
        print(f"Input: {test_case['text']}")

        try:
            # Perform redaction
            result = model.redact_text(test_case["text"])

            print(f"Redacted: {result.redacted_text}")
            print(
                f"Sensitivity: {result.sensitivity_level} (confidence: {result.confidence:.2f})"
            )
            print(f"Entities found: {list(result.detection_summary.keys())}")
            print(f"Methods used: {result.method_summary}")

            # Check if we found expected entities
            found_entities = set(result.detection_summary.keys())
            expected_entities = set(test_case["expected_entities"])

            if found_entities.intersection(expected_entities):
                print("✅ Test passed - Found some expected entities")
                success_count += 1
            else:
                print("⚠️  Test warning - No expected entities found")
                print(f"   Expected: {expected_entities}")
                print(f"   Found: {found_entities}")

        except Exception as e:
            print(f"❌ Test failed with error: {e}")

    print(f"\n📊 Results: {success_count}/{len(test_cases)} tests passed")
    return success_count == len(test_cases)


def test_sensitivity_levels():
    """Test sensitivity level filtering"""
    print("\n🔒 Testing Sensitivity Level Filtering")
    print("=" * 50)

    model = EnhancedRedactionModel()

    test_text = "John Smith's email john.smith@ing.com and account NL91ABNA0417164300 with balance EUR 10,000"

    levels = ["C1", "C2", "C3", "C4"]

    for level in levels:
        print(f"\n🎯 Testing level {level}:")
        result = model.redact_text(test_text, sensitivity_level=level)
        print(f"   Original: {test_text}")
        print(f"   Redacted: {result.redacted_text}")
        print(f"   Entities: {result.total_redacted}")


def test_performance():
    """Test performance with larger text"""
    print("\n⚡ Testing Performance")
    print("=" * 50)

    model = EnhancedRedactionModel()

    # Create a larger test text
    large_text = """
    Dear Customer,
    
    We are writing to inform you about recent transactions on your account NL91ABNA0417164300.
    
    Transaction Details:
    - Transaction ID: 550e8400-e29b-41d4-a716-446655440000
    - Amount: EUR 5,234.56
    - Date: 2025-01-15
    - Recipient: John Smith (john.smith@ing.com)
    - Phone: +31 20 123 4567
    
    Additional transactions:
    - Transaction ID: 651f9511-f30c-52e5-b827-557655551111
    - Amount: EUR 1,500.00
    - Date: 2025-01-16
    
    Your current balance is EUR 15,734.56.
    
    If you have any questions, please contact our customer service at support@ing.com
    or call us at +31 20 555 0123.
    
    Best regards,
    ING Bank Customer Service
    Employee ID: CS123456
    """

    import time

    start_time = time.time()

    result = model.redact_text(large_text)

    end_time = time.time()
    processing_time = end_time - start_time

    print(f"Processing time: {processing_time:.3f} seconds")
    print(f"Text length: {len(large_text)} characters")
    print(f"Entities found: {result.total_redacted}")
    print(f"Detection summary: {result.detection_summary}")
    print(f"Method summary: {result.method_summary}")


def main():
    """Run all tests"""
    print("🚀 Enhanced Redaction Model Test Suite")
    print("=" * 60)

    # Check dependencies
    print("📦 Checking dependencies...")
    try:
        import numpy
        import pandas
        import sklearn

        print("✅ Core ML libraries available")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Install with: pip install numpy pandas scikit-learn")
        return

    try:
        import spacy

        print("✅ spaCy available")
    except ImportError:
        print("⚠️  spaCy not available - NER features will be limited")
        print(
            "Install with: pip install spacy && python -m spacy download en_core_web_sm"
        )

    # Run tests
    try:
        test_basic_redaction()
        test_sensitivity_levels()
        test_performance()

        print("\n🎉 All tests completed!")
        print("\n💡 Next steps:")
        print(
            "1. Install spaCy for better NER: pip install spacy && python -m spacy download en_core_web_sm"
        )
        print(
            "2. Generate training data: python synthetic_data/generate_synthetic_data.py"
        )
        print("3. Train the model: python train_enhanced_model.py")
        print("4. Integrate with API: Update src/sensitivity_classifier.py")

    except Exception as e:
        print(f"\n💥 Test suite failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
