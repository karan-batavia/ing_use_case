# 🏦 Enhanced Banking Redaction Model - Usage Guide

Your enhanced redaction model is now trained and ready to use! Here are all the ways you can use it:

## 🚀 Quick Start

### 1. **Test Your Model Right Now**
```bash
python use_trained_model.py
```

This will:
- ✅ Load your trained model
- 🧪 Run test cases on banking data
- 🎮 Offer interactive mode to test your own text

**Expected Output:**
```
🧪 Testing Enhanced Redaction Model
========================================
📝 Test 1: Customer banking email
📊 Sensitivity Level: C3
📄 Original: Dear John Smith, your account NL91ABNA0417164300 has a balance of €50,000...
🔒 Redacted: Dear <PERSON_NAME>, your account <IBAN> has a balance of <AMOUNT>...
🎯 Entities Found: 4 entities
📈 Confidence: 0.892
```

---

## 🔧 Integration Options

### 2. **API Integration** (Recommended for Production)
```bash
# Install FastAPI if needed
pip install fastapi uvicorn

# Run the API server
uvicorn api_integration:app --reload --port 8000
```

**API Endpoints:**
- `POST /redact` - Redact text
- `GET /health` - Health check
- `GET /test` - Quick test
- `GET /docs` - Interactive API docs

**Example API Call:**
```bash
curl -X POST "http://localhost:8000/redact" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Employee John Smith transferred €50,000 to NL91ABNA0417164300",
    "sensitivity_level": "C4",
    "include_details": true
  }'
```

**Response:**
```json
{
  "redacted_text": "Employee <PERSON_NAME> transferred <AMOUNT> to <IBAN>",
  "entities_detected": 3,
  "confidence": 0.91,
  "processing_time_ms": 45.2,
  "details": {
    "detection_summary": {"PERSON_NAME": 1, "AMOUNT": 1, "IBAN": 1}
  }
}
```

### 3. **Batch Processing** (For Large Datasets)

**Process CSV files:**
```bash
python batch_processing.py \
  --mode csv \
  --input your_data.csv \
  --output redacted_data.csv \
  --text-column "customer_message" \
  --sensitivity C3
```

**Process entire directories:**
```bash
python batch_processing.py \
  --mode directory \
  --input ./documents \
  --output ./redacted_documents \
  --sensitivity C4
```

**Test mode (quick check):**
```bash
python batch_processing.py --mode test --sensitivity C3
```

---

## 📊 Sensitivity Levels

| Level | Description | Use Case | Redaction Scope |
|-------|-------------|----------|-----------------|
| **C1** | Public Info | Marketing, Public docs | Minimal (PII only) |
| **C2** | Internal | Employee docs, Procedures | Moderate |
| **C3** | Confidential | Customer data, Transactions | High |
| **C4** | Restricted | Sensitive financial data | Maximum |

---

## 💻 Programmatic Usage

### In Your Python Code:
```python
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.enhanced_redaction_model_clean import create_enhanced_redaction_model

# Load model
model = create_enhanced_redaction_model()

# Redact text
result = model.redact_text(
    text="John Smith works at ING Bank, account NL91ABNA0417164300",
    sensitivity_level="C3"
)

print(f"Original: {result.original_text}")
print(f"Redacted: {result.redacted_text}")
print(f"Entities: {len(result.detections)}")
print(f"Confidence: {result.confidence}")
```

### Access Detailed Results:
```python
# Get detailed entity information
for detection in result.detections:
    print(f"Found: {detection.text} -> {detection.label}")
    print(f"Method: {detection.detection_method}")
    print(f"Confidence: {detection.confidence}")

# Get summaries
print(f"Entity types: {result.detection_summary}")
print(f"Methods used: {result.method_summary}")
```

---

## 🔧 Advanced Configuration

### Custom Sensitivity Rules:
```python
# Create model with custom config
from src.enhanced_redaction_model_clean import EnhancedRedactionModel

model = EnhancedRedactionModel()

# Override sensitivity filtering
result = model.redact_text(
    text="Your text here",
    sensitivity_level="C3",
    # Additional options available in the model
)
```

### Performance Optimization:
```python
# For batch processing, load model once
model = create_enhanced_redaction_model()

# Process multiple texts efficiently
texts = ["text1", "text2", "text3"]
results = []

for text in texts:
    result = model.redact_text(text, "C3")
    results.append(result)
```

---

## 📈 Monitoring & Evaluation

### Check Model Performance:
```python
# Review detection summary
print(f"Detection methods: {result.method_summary}")
# Example: {'regex': 5, 'ner': 3, 'hybrid': 2}

# Check confidence scores
if result.confidence < 0.7:
    print("⚠️ Low confidence - manual review recommended")
```

### Entity Coverage:
The model detects:
- 👤 **Personal**: Names, emails, phone numbers
- 💰 **Financial**: IBANs, amounts, account numbers
- 🏢 **Corporate**: Employee IDs, company data
- 📍 **Geographic**: Addresses, postal codes
- 🆔 **Identifiers**: Transaction IDs, reference numbers

---

## 🚨 Error Handling

```python
try:
    result = model.redact_text(text, sensitivity_level)
    if result.confidence < 0.5:
        logger.warning(f"Low confidence redaction: {result.confidence}")
except Exception as e:
    logger.error(f"Redaction failed: {e}")
    # Fallback to basic redaction or error handling
```

---

## 📁 File Structure

After training, you'll have:
```
models/
├── enhanced_redaction_model.joblib    # Trained model
├── vectorizer.joblib                  # Text vectorizer
├── training_metrics.json              # Performance metrics
└── model_config.json                  # Configuration

src/
├── enhanced_redaction_model_clean.py  # Core model
└── presidio.py                        # Legacy fallback

# Usage examples
├── use_trained_model.py               # Quick testing
├── api_integration.py                 # REST API
└── batch_processing.py                # Batch processing
```

---

## 🎯 Next Steps

1. **Start with**: `python use_trained_model.py`
2. **For production**: Set up the API with `uvicorn api_integration:app`
3. **For large datasets**: Use `batch_processing.py`
4. **Integration**: Use the programmatic interface in your existing code

Your enhanced model combines:
- 🧠 **NER** (spaCy) for contextual understanding
- 🔍 **Regex** patterns for banking-specific entities
- 🎯 **ML Classification** for sensitivity-based filtering
- ⚡ **High Performance** with confidence scoring

**Ready to redact! 🚀**