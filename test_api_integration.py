#!/usr/bin/env python3
"""
Test script for the enhanced API integration
Tests the new redaction endpoints with enhanced model
"""

import requests
import json
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.append(str(project_root))


def test_api_integration():
    """Test the enhanced redaction API"""

    base_url = "http://localhost:8000"

    print("🔧 Testing Enhanced Redaction API Integration")
    print("=" * 50)

    # Test 1: Get redaction info (no auth required)
    print("\n1️⃣ Testing /redact/info endpoint...")
    try:
        response = requests.get(f"{base_url}/redact/info")
        if response.status_code == 200:
            info = response.json()
            print("✅ API info retrieved successfully:")
            print(f"   Enhanced model available: {info['enhanced_model_available']}")
            print(f"   Status: {info['status']}")
            print(f"   Available methods: {list(info['methods'].keys())}")
            print(f"   Sensitivity levels: {list(info['sensitivity_levels'].keys())}")
        else:
            print(f"❌ Failed to get API info: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error testing info endpoint: {e}")
        return False

    # Test 2: Test redaction endpoint (this will fail without auth, but we can check the structure)
    print("\n2️⃣ Testing /redact endpoint structure...")
    test_data = {
        "text": "John Smith works at ING bank. His email is john.smith@ing.com and IBAN is NL91ABNA0417164300.",
        "sensitivity_level": "C3",
        "method": "auto",
    }

    try:
        response = requests.post(f"{base_url}/redact", json=test_data)
        if response.status_code == 401:
            print("✅ Endpoint exists and requires authentication (as expected)")
            print("   Response:", response.json())
        elif response.status_code == 200:
            result = response.json()
            print("✅ Redaction successful:")
            print(f"   Original: {result['original_text']}")
            print(f"   Redacted: {result['redacted_text']}")
            print(f"   Method used: {result['method_used']}")
            print(f"   Entities found: {result['entities_found']}")
        else:
            print(f"⚠️  Unexpected response: {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"❌ Error testing redact endpoint: {e}")
        return False

    print("\n✅ API integration test completed!")
    print("\n📋 Next Steps:")
    print("1. Start the API server: uvicorn api:app --reload --port 8000")
    print("2. Visit http://localhost:8000/docs for interactive API documentation")
    print("3. Test with authentication using the /auth endpoints")

    return True


def test_local_model():
    """Test the enhanced model locally (without API)"""
    print("\n🧪 Testing Enhanced Model Locally")
    print("=" * 40)

    try:
        from src.enhanced_redaction_model_clean import create_enhanced_redaction_model

        print("Loading enhanced model...")
        model = create_enhanced_redaction_model()

        # Test redaction
        test_text = "John Smith works at ING bank. His email is john.smith@ing.com and IBAN is NL91ABNA0417164300."

        print(f"\nOriginal text: {test_text}")

        for sensitivity in ["C1", "C2", "C3", "C4"]:
            result = model.redact_text(test_text, sensitivity_level=sensitivity)
            print(f"\n{sensitivity} Redaction:")
            print(f"  Redacted: {result.redacted_text}")
            print(f"  Confidence: {result.confidence:.3f}")
            print(f"  Entities: {[d.label for d in result.detections]}")

        print("\n✅ Local model test completed!")
        return True

    except Exception as e:
        print(f"❌ Error testing local model: {e}")
        return False


if __name__ == "__main__":
    success = True

    # Test local model first
    success &= test_local_model()

    print("\n" + "=" * 60)

    # Test API integration
    success &= test_api_integration()

    if success:
        print("\n🎉 All tests completed! Your enhanced redaction model is ready!")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
