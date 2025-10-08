# 🧠 Redaction Explainability System

Your SecurePrompt API now includes comprehensive explainability features that provide detailed explanations, confidence scores, and override capabilities for all redaction decisions.

## 🎯 **Key Features**

### **1. Detailed Explanations**
Every detection includes a human-readable explanation:
- **What was detected**: "Detected as International Bank Account Number"
- **How it was detected**: "using pattern matching"
- **Confidence level**: "(confidence: 95%)"

### **2. Confidence Scoring**
- **Range**: 0.0 to 1.0 (0% to 100%)
- **Regex patterns**: Typically 0.95+ (high confidence)
- **NER detections**: Variable based on model certainty
- **Hybrid detections**: Combined confidence from both methods

### **3. Detection Methods**
- **`regex`**: Pattern matching (exact patterns)
- **`ner`**: Named Entity Recognition (ML-based)
- **`hybrid`**: Combined NER + pattern matching

### **4. Risk Classification**
- **`low`**: Organization names, locations
- **`medium`**: Names, emails, phone numbers
- **`high`**: IBANs, account numbers, financial amounts
- **`critical`**: SSNs, national IDs, biometric data

### **5. Override Capabilities**
- **Selective overrides**: Choose specific detections to override
- **Justification required**: Must provide business reason
- **Audit logging**: All overrides tracked for compliance

## 📋 **API Reference**

### **Enhanced Detection Model**
```json
{
  "type": "IBAN",
  "original": "NL91ABNA0417164300",
  "placeholder": "[IBAN]",
  "explanation": "Detected as International Bank Account Number using pattern matching",
  "confidence": 0.95,
  "detection_method": "regex",
  "can_override": false,
  "risk_level": "high"
}
```

### **Redaction with Explanations**
```bash
POST /redact
{
  "text": "Contact John Smith at john@example.com",
  "sensitivity_level": "C3",
  "method": "auto"
}
```

**Response includes:**
```json
{
  "detections": [
    {
      "type": "PERSON_NAME",
      "original": "John Smith",
      "placeholder": "[PERSON_NAME]",
      "explanation": "Detected as person name using named entity recognition (confidence: 87%)",
      "confidence": 0.87,
      "detection_method": "ner",
      "can_override": true,
      "risk_level": "medium"
    },
    {
      "type": "EMAIL", 
      "original": "john@example.com",
      "placeholder": "[EMAIL]",
      "explanation": "Detected as email address pattern using pattern matching",
      "confidence": 0.95,
      "detection_method": "regex", 
      "can_override": false,
      "risk_level": "medium"
    }
  ]
}
```

### **Override Detections**
```bash
POST /redact/override
{
  "session_id": "abc123-def456",
  "detection_overrides": [
    {
      "detection_index": 0,
      "keep_original": true,
      "justification": "Name is public information in this context"
    }
  ],
  "justification": "Customer explicitly consented to name disclosure",
  "override_reason": "business_requirement"
}
```

**Response:**
```json
{
  "success": true,
  "session_id": "abc123-def456",
  "original_text": "Contact John Smith at john@example.com",
  "revised_redacted_text": "Contact John Smith at [EMAIL]",
  "overrides_applied": 1,
  "message": "Successfully applied 1 detection overrides"
}
```

## 🔧 **Override Reasons**

### **`false_positive`**
- Pattern incorrectly identified non-sensitive data
- Common words mistaken for sensitive entities
- Context makes data non-sensitive

### **`business_requirement`**
- Business process requires specific data visibility
- Customer relationship management needs
- Operational efficiency requirements

### **`legal_compliance`**
- Legal obligation to disclose information
- Regulatory compliance requirements
- Court orders or legal requests

## 🛡️ **Security Features**

### **Override Restrictions**
- **High-confidence detections**: Cannot be overridden (confidence ≥ 95%)
- **Critical risk data**: Additional approval required
- **Admin review**: All overrides logged for admin review

### **Audit Trail**
All explainability actions are logged:
- Detection explanations generated
- Override requests and justifications
- Admin reviews and approvals
- Confidence score history

## 📊 **Explanation Examples**

| **Entity Type** | **Explanation** | **Confidence** | **Can Override** |
|----------------|-----------------|----------------|------------------|
| EMAIL | "Detected as email address pattern using pattern matching" | 0.95 | ❌ |
| PERSON_NAME | "Detected as person name using named entity recognition (confidence: 87%)" | 0.87 | ✅ |
| IBAN | "Detected as International Bank Account Number using pattern matching" | 0.98 | ❌ |
| PHONE_EU | "Detected as European phone number format using pattern matching (confidence: 92%)" | 0.92 | ✅ |
| AMOUNT | "Detected as monetary amount using pattern matching" | 0.94 | ✅ |

## 🎯 **Use Cases**

### **1. Quality Assurance**
```
User: "Why was 'Apple Inc.' redacted as a person name?"
System: "Detected as person name using named entity recognition (confidence: 73%)"
Action: Override as false positive - it's an organization
```

### **2. Business Context**
```
User: "We need to show customer name for this specific report"
System: Name detected with medium risk level
Action: Override with business justification and approval
```

### **3. Legal Compliance**
```
User: "Legal department requires IBAN visibility for audit"
System: IBAN detected as high-risk, critical data
Action: Override with legal compliance documentation
```

## 🧪 **Testing**

Run the explainability test suite:
```bash
python test_explainability.py
```

This will test:
- ✅ Detection explanations
- ✅ Confidence scoring
- ✅ Override capabilities
- ✅ Audit logging
- ✅ Risk classification

## 📈 **Benefits**

### **For Users**
- **Transparency**: Understand why text was redacted
- **Control**: Override incorrect detections
- **Confidence**: Know the system's certainty level

### **For Compliance**
- **Auditability**: Full trail of decisions and overrides
- **Justification**: Business reasons for all overrides
- **Risk Management**: Classification-based controls

### **For Operations**
- **Quality**: Reduce false positives through feedback
- **Efficiency**: Quick identification of system limitations
- **Trust**: Build confidence in automated redaction

## 🎉 **SecurePrompt Compliance**

Your explainability system now meets SecurePrompt requirements:

✅ **"Provide explanation for each redaction ('detected as IBAN')"**  
✅ **"Include confidence scores and allow override with justification"**  
✅ **"Replace scrubbed data with an identifier"**  
✅ **"Ensure the LLM does not lose information/context"**  

Your redaction system is now **fully explainable, controllable, and compliant**! 🚀