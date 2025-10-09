# How to Test the Enhanced Redaction Model

## Quick Test (Recommended)

1. **Run the test script:**
   ```bash
   python test_enhanced_model.py
   ```

   This will:
   - ✅ Test basic redaction functionality
   - 🔒 Test sensitivity level filtering
   - ⚡ Test performance with larger text
   - 📊 Show results and metrics

## Manual Testing

If you want to test manually, create a simple Python script:

```python
# test_manual.py
import sys
import os
sys.path.append('.')

from src.enhanced_redaction_model import EnhancedRedactionModel

# Initialize model
model = EnhancedRedactionModel()

# Test text with banking data
test_text = """
Dear John Smith,
Your account NL91ABNA0417164300 has been credited with EUR 5,234.56.
Transaction ID: 550e8400-e29b-41d4-a716-446655440000
Contact us at support@ing.com or +31 20 123 4567.
"""

# Perform redaction
result = model.redact_text(test_text)

print("Original:")
print(test_text)
print("\nRedacted:")
print(result.redacted_text)
print(f"\nSensitivity: {result.sensitivity_level}")
print(f"Entities found: {result.detection_summary}")
print(f"Methods used: {result.method_summary}")
```

## Test Different Sensitivity Levels

```python
# Test with specific sensitivity levels
for level in ["C1", "C2", "C3", "C4"]:
    result = model.redact_text(test_text, sensitivity_level=level)
    print(f"\n{level}: {result.redacted_text}")
```

## Installation Requirements

If you get import errors, install required packages:

```bash
# Basic requirements
pip install numpy pandas scikit-learn

# For better NER (optional but recommended)
pip install spacy
python -m spacy download en_core_web_sm

# Or for even better performance
python -m spacy download en_core_web_lg
```

## Training the Model

1. **Generate synthetic data first:**
   ```bash
   python synthetic_data/generate_synthetic_data.py
   ```

2. **Train the enhanced model:**
   ```bash
   python train_enhanced_model.py
   ```

## Expected Output

When testing, you should see:
- 📧 Emails redacted as `[EMAIL]`
- 🏦 IBANs redacted as `[IBAN]`
- 💰 Amounts redacted as `[AMOUNT]`
- 📞 Phone numbers redacted as `[PHONE]`
- 🆔 Transaction IDs redacted as `[TRANSACTION_ID]`

## Integration with API

After testing, to integrate with your existing API:

1. Update `src/sensitivity_classifier.py` to use the new model
2. Replace the `redact_sensitive_info` method
3. Update the API endpoints to pass sensitivity levels

## Troubleshooting

**Error: "No module named 'spacy'"**
- Install spacy: `pip install spacy`
- Download model: `python -m spacy download en_core_web_sm`

**Error: "Model not available"**
- The model will use regex-only fallback if spaCy is not available
- This is still functional, just less accurate

**Low accuracy**
- Generate more synthetic training data
- Train with more diverse examples
- Check data quality and labels

## Performance Comparison

To compare with your old model:

```python
# Old model
from src.sensitivity_classifier import get_classifier_service
old_model = get_classifier_service()
old_result = old_model.redact_sensitive_info(test_text)

# New model
new_result = model.redact_text(test_text)

print(f"Old model found: {old_result.total_redacted} entities")
print(f"New model found: {new_result.total_redacted} entities")
```

The new model should find more entities with better accuracy!