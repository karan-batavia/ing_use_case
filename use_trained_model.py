#!/usr/bin/env python3
"""
Quick example of how to use your trained enhanced redaction model
"""

import sys
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from src.enhanced_redaction_model_clean import create_enhanced_redaction_model
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you've trained the model first!")
    sys.exit(1)


def load_trained_model():
    """Load the trained model"""
    print("🔄 Loading trained enhanced redaction model...")

    # Create model instance
    model = create_enhanced_redaction_model()

    # Try to load trained weights (if saved)
    model_dir = Path("models")
    if model_dir.exists():
        print(f"✅ Found models directory: {model_dir}")
        # The model should auto-load if weights are saved
    else:
        print("⚠️  No models directory found - using freshly initialized model")

    return model


def test_redaction_examples():
    """Test the model with various banking examples"""

    # Load the model
    model = load_trained_model()

    # Test examples
    test_cases = [
        {
            "text": "Dear John Smith, your account NL91ABNA0417164300 has a balance of €50,000. Contact us at support@ingbank.nl or call +31-20-123-4567.",
            "sensitivity": "C3",
            "description": "Customer banking email",
        },
        {
            "text": "Employee ID: EMP12345, Name: Sarah Johnson, Phone: +31-6-87654321, Email: s.johnson@ing.com, Department: Risk Management",
            "sensitivity": "C2",
            "description": "Employee data",
        },
        {
            "text": "Transaction: Transfer €25,000 from NL91ABNA0417164300 to DE89370400440532013000 on 2024-01-15. Reference: TXN789456123",
            "sensitivity": "C4",
            "description": "High-value transaction",
        },
        {
            "text": "Our new banking product offers competitive rates. Visit www.ing.nl for more information.",
            "sensitivity": "C1",
            "description": "Public marketing content",
        },
    ]

    print("\n🧪 Testing Enhanced Redaction Model")
    print("=" * 60)

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📝 Test {i}: {test_case['description']}")
        print(f"📊 Sensitivity Level: {test_case['sensitivity']}")
        print(f"📄 Original: {test_case['text']}")

        try:
            # Perform redaction
            result = model.redact_text(
                text=test_case["text"], sensitivity_level=test_case["sensitivity"]
            )

            print(f"🔒 Redacted: {result.redacted_text}")
            print(f"🎯 Entities Found: {len(result.detections)} entities")
            print(f"📈 Confidence: {result.confidence:.3f}")
            print(f"🔧 Detection Methods: {result.method_summary}")
            print(f"📊 Entity Types: {result.detection_summary}")

        except Exception as e:
            print(f"❌ Error: {e}")

        print("-" * 40)


def interactive_redaction():
    """Interactive redaction session"""

    model = load_trained_model()

    print("\n🎮 Interactive Redaction Mode")
    print("=" * 40)
    print("Enter text to redact (or 'quit' to exit)")
    print("Sensitivity levels: C1 (low) -> C4 (high)")

    while True:
        print("\n" + "─" * 40)
        text = input("📝 Enter text: ").strip()

        if text.lower() in ["quit", "exit", "q"]:
            print("👋 Goodbye!")
            break

        if not text:
            continue

        sensitivity = (
            input("📊 Sensitivity level (C1/C2/C3/C4) [C3]: ").strip().upper() or "C3"
        )

        if sensitivity not in ["C1", "C2", "C3", "C4"]:
            sensitivity = "C3"
            print(f"⚠️  Invalid level, using {sensitivity}")

        try:
            result = model.redact_text(text=text, sensitivity_level=sensitivity)

            print(f"\n🔒 Result:")
            print(f"  Original:  {text}")
            print(f"  Redacted:  {result.redacted_text}")
            print(f"  Entities:  {len(result.detections)} found")
            print(f"  Methods:   {result.method_summary}")
            print(f"  Types:     {result.detection_summary}")

        except Exception as e:
            print(f"❌ Error: {e}")


def main():
    """Main function"""
    print("🏦 Enhanced Banking Redaction Model - Usage Example")
    print("=" * 60)

    # Run test examples
    test_redaction_examples()

    # Ask if user wants interactive mode
    print("\n" + "=" * 60)
    choice = input("🎮 Would you like to try interactive mode? (y/N): ").strip().lower()

    if choice in ["y", "yes"]:
        interactive_redaction()
    else:
        print(
            "✅ Demo completed! Use this script as a template for your own applications."
        )


if __name__ == "__main__":
    main()
