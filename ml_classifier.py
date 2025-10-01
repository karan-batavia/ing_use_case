import json
import random
import os
from typing import List, Dict, Tuple # FIXED: Tuple imported
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
import spacy

# Import necessary functions/data from the existing setup file
# NOTE: The imported 'label_prompt' function is assumed to use the other imported keywords and objects
from ml_setup import (
    label_prompt, SENSITIVITY_MAP, nlp, CONTEXT_WINDOW,
    RELIGION_KEYWORDS, ETHNICITY_KEYWORDS, SEXUAL_ORIENTATION_KEYWORDS,
    POLITICAL_KEYWORDS, HEALTH_KEYWORDS, TRADE_UNION_KEYWORDS
)

# ==============================================================================
# 1. DATA LOADING (Robust Mock Data)
# ==============================================================================

def get_robust_mock_data() -> List[Dict]:
    """
    Generates a robust mock dataset with examples covering all sensitivity levels (C0 to C4).
    This dataset replaces the original small mock data for better testing.
    """
    return [
        # C4 EXAMPLES (Highest Risk - IBAN, SSN, Sensitive Health/Political)
        {'text': 'The transfer failed for account IBAN BE68539007547034. He is a card-carrying member of the far-right party.',
         'entities': [{'entity': 'BE68539007547034', 'type': 'iban', 'sensitivity': 'C4'},
                      {'entity': 'far-right', 'type': 'political_opinion', 'sensitivity': 'C4'}]}, # Max Risk C4
        {'text': 'My SSN is 901231-123.45. This patient has advanced AIDS.',
         'entities': [{'entity': '901231-123.45', 'type': 'social_security', 'sensitivity': 'C4'},
                      {'entity': 'AIDS', 'type': 'health_condition', 'sensitivity': 'C4'}]}, # Max Risk C4
        {'text': 'I need the PIN 12345 for the credit card transaction.',
         'entities': [{'entity': '12345', 'type': 'pin', 'sensitivity': 'C4'},
                      {'entity': 'credit card', 'type': 'credit_card', 'sensitivity': 'C4'}]}, # Max Risk C4
        {'text': 'Transaction ID: 0101-TX-999. They converted to Islam last year.',
         'entities': [{'entity': '0101-TX-999', 'type': 'transaction', 'sensitivity': 'C4'},
                      {'entity': 'Islam', 'type': 'religion', 'sensitivity': 'C4'}]}, # Max Risk C4

        # C3 EXAMPLES (High Risk - Email, Customer ID, Address, DOB)
        {'text': 'Client CUST-1025, born on 15/03/1985, lives at 1000 Brussels.',
         'entities': [{'entity': 'CUST-1025', 'type': 'customer_number', 'sensitivity': 'C3'},
                      {'entity': '15/03/1985', 'type': 'date_of_birth', 'sensitivity': 'C3'},
                      {'entity': '1000 Brussels', 'type': 'postal_code_city', 'sensitivity': 'C3'}]}, # Max Risk C3
        {'text': 'The email for the account owner is jane.doe@corp.com.',
         'entities': [{'entity': 'jane.doe@corp.com', 'type': 'email', 'sensitivity': 'C3'}]}, # Max Risk C3
        {'text': 'Please send the documents to John Smith, ID: BEL-999-XYZ.',
         'entities': [{'entity': 'John Smith', 'type': 'name', 'sensitivity': 'C3'},
                      {'entity': 'BEL-999-XYZ', 'type': 'belgian_id', 'sensitivity': 'C3'}]}, # Max Risk C3
        {'text': 'Their contact phone number is +32 2 123 4567.',
         'entities': [{'entity': '+32 2 123 4567', 'type': 'phone', 'sensitivity': 'C4'}]}, # Max Risk C4 (Phone is C4 in SENSITIVITY_MAP)

        # C2 EXAMPLES (Medium Risk - General NER)
        {'text': 'Apple is working on a new product in London next year.',
         'entities': [{'entity': 'Apple', 'type': 'ORG', 'sensitivity': 'C2'},
                      {'entity': 'London', 'type': 'GPE', 'sensitivity': 'C2'},
                      {'entity': 'next year', 'type': 'DATE', 'sensitivity': 'C2'}]}, # Max Risk C2
        {'text': 'The meeting with Microsoft in Paris is scheduled for Tuesday.',
         'entities': [{'entity': 'Microsoft', 'type': 'ORG', 'sensitivity': 'C2'},
                      {'entity': 'Paris', 'type': 'GPE', 'sensitivity': 'C2'}]}, # Max Risk C2
        {'text': 'Mr. Jones reviewed the documents on November 10th.',
         'entities': [{'entity': 'Jones', 'type': 'PERSON', 'sensitivity': 'C2'},
                      {'entity': 'November 10th', 'type': 'DATE', 'sensitivity': 'C2'}]}, # Max Risk C2
        {'text': 'The budget for Q3 is around fifty thousand Euros.',
         'entities': [{'entity': 'Q3', 'type': 'DATE', 'sensitivity': 'C2'},
                      {'entity': 'fifty thousand', 'type': 'MONEY', 'sensitivity': 'C2'}]}, # Max Risk C2

        # C1 EXAMPLES (Low Risk - Years, Document Types)
        {'text': 'The final report for the 2024 project is complete.',
         'entities': [{'entity': '2024', 'type': 'year', 'sensitivity': 'C1'},
                      {'entity': 'report', 'type': 'document_type', 'sensitivity': 'C1'}]}, # Max Risk C1
        {'text': 'Create a new template for the annual review 2025.',
         'entities': [{'entity': '2025', 'type': 'year', 'sensitivity': 'C1'},
                      {'entity': 'review', 'type': 'document_type', 'sensitivity': 'C1'}]}, # Max Risk C1
        {'text': 'A link to the public announcement is on the website.',
         'entities': [{'entity': 'link', 'type': 'link', 'sensitivity': 'C1'}]}, # Max Risk C1
        {'text': 'The investment period runs from 2020-2025.',
         'entities': [{'entity': '2020-2025', 'type': 'year_range', 'sensitivity': 'C1'}]}, # Max Risk C1

        # C0 EXAMPLES (No Identified Risk)
        {'text': 'Please draft a new email for external communication.', 'entities': []}, # Max Risk C0
        {'text': 'What is the current status of the global market?', 'entities': []}, # Max Risk C0
        {'text': 'We need to discuss project scope during the next call.', 'entities': []}, # Max Risk C0
        {'text': 'This is a test sentence with no sensitive data.', 'entities': []}, # Max Risk C0
    ]

def load_labeled_data(file_path: str) -> List[Dict]:
    """Loads labeled data from a JSON file, falling back to robust mock data."""
    try:
        # In a real scenario, we would try to load the file
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: Labeled data file not found at {file_path}.")
        print("Using robust mock data for demonstration purposes.")
        return get_robust_mock_data()

# ==============================================================================
# 2. FEATURE EXTRACTION (Rule-based + ML Features)
# ... (extract_features_for_ml remains the same)
# ==============================================================================

def extract_features_for_ml(text: str) -> Dict:
    """
    Extracts a feature vector for a given text input.
    This combines the existing rule-based detection counts (C1-C4)
    with basic NLP/ML features like length and keyword counts.
    """
    # Use the existing labeling logic to get rule-based features
    # NOTE: 'label_prompt' is imported from ml_setup.py
    # NOTE: Set use_c1_c2=False to avoid dependency on external data_dir logic
    labeled_result = label_prompt(text, inject=False, use_spacy=True, use_c1_c2=False) 
    stats = labeled_result["statistics"]

    features = {
        # Rule-based (counts of detected entities)
        'c4_count': stats.get('c4_count', 0),
        'c3_count': stats.get('c3_count', 0),
        'c2_count': stats.get('c2_count', 0),
        'c1_count': stats.get('c1_count', 0),
        'total_entities': stats.get('total_entities', 0),
        'spacy_ner_count': stats.get('spacy_detected', 0),
        
        # NLP Features
        'text_length': len(text),
        'word_count': len(text.split()),
        'has_question_mark': '?' in text,
        
        # Contextual Keyword Flags (C4 special categories)
        'has_religion_keyword': any(k in text.lower() for k in RELIGION_KEYWORDS),
        'has_ethnicity_keyword': any(k in text.lower() for k in ETHNICITY_KEYWORDS),
        'has_sexual_orientation_keyword': any(k in text.lower() for k in SEXUAL_ORIENTATION_KEYWORDS),
        'has_political_keyword': any(k in text.lower() for k in POLITICAL_KEYWORDS),
        'has_health_keyword': any(k in text.lower() for k in HEALTH_KEYWORDS),
    }
    
    return features


def create_ml_dataset(labeled_data: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts labeled data into ML feature matrix (X) and target vector (y).
    The target variable 'y' is the highest sensitivity level present in the prompt.
    """
    X = []
    y = []
    
    # Sensitivity mapping for target variable: C4=4, C3=3, C2=2, C1=1, None=0
    risk_level_map = {'C4': 4, 'C3': 3, 'C2': 2, 'C1': 1}
    
    for item in labeled_data:
        text = item['text']
        features = extract_features_for_ml(text)
        X.append(list(features.values())) # Feature vector
        
        # Determine the highest sensitivity in the prompt
        highest_risk = 0
        for entity in item.get('entities', []):
            sensitivity = entity.get('sensitivity', 'C0')
            highest_risk = max(highest_risk, risk_level_map.get(sensitivity, 0))
        
        y.append(highest_risk) # Target variable (highest risk level)

    return np.array(X), np.array(y)

# ==============================================================================
# 3. TRAIN SIMPLE CLASSIFIER (Logistic Regression)
# ==============================================================================

def train_simple_classifier(X: np.ndarray, y: np.ndarray):
    """
    Trains a simple Logistic Regression classifier based on the extracted features.
    """
    if X.shape[0] < 2:
        print("Not enough data to train the classifier.")
        return None

    # Split data using stratification (possible now due to robust mock data)
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    except ValueError as e:
        print(f"Warning: Stratification failed ({e}). Falling back to simple random split.")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train model (using a simple Linear Classifier like Logistic Regression)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    # Evaluation (simple for demo)
    accuracy = model.score(X_test, y_test)
    print(f"\nSimple Classifier (Logistic Regression) Trained.")
    print(f"Test Accuracy: {accuracy:.4f}")
    
    return model

# ==============================================================================
# 4. BERT-BASED CLASSIFIER (Outline)
# ... (outline_bert_classifier remains the same)
# ==============================================================================

def outline_bert_classifier():
    """
    Outlines the steps required to train a BERT-based classifier.
    This is conceptual as it requires large libraries (Hugging Face, PyTorch/TensorFlow).
    """
    print("\n" + "="*50)
    print("CONCEPTUAL OUTLINE: BERT-BASED SENSITIVITY CLASSIFIER")
    print("="*50)
    
    print("1. Data Preparation:")
    print("   - Prompts must be prepared with the overall sensitivity label (C1, C2, C3, C4).")
    print("   - Tokenization: Use a pre-trained BERT tokenizer (e.g., from `transformers` library) to convert text into input IDs and attention masks.")
    
    print("\n2. Model Selection and Loading:")
    print("   - Load a pre-trained BERT model (e.g., 'bert-base-uncased') for sequence classification (AutoModelForSequenceClassification).")
    
    print("\n3. Training Loop:")
    print("   - Split the tokenized data into training and validation sets.")
    print("   - Define training arguments (epochs, batch size, learning rate).")
    print("   - Use the `Trainer` class (Hugging Face) or manually implement the PyTorch/TensorFlow training loop.")
    print("   - The model learns to classify the entire prompt based on its text content into the highest risk category (C4, C3, C2, C1).")
    
    print("\n4. Evaluation and Prediction:")
    print("   - Evaluate the fine-tuned model on the test set using metrics like F1-score and accuracy.")
    print("   - Use the fine-tuned model to predict the overall risk level for new, unseen prompts.")
    print("\nNote: This requires heavy dependencies (transformers, torch/tensorflow) and labeled data.")


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    MOCK_LABELED_FILE = "labeled_data/labeled_prompts_hybrid.json" 
    
    # Load or create robust mock data
    labeled_data = load_labeled_data(MOCK_LABELED_FILE)

    if labeled_data:
        # Create the feature matrix (X) and target vector (y)
        X, y = create_ml_dataset(labeled_data)
        
        # Train a simple classifier
        classifier_model = train_simple_classifier(X, y)
        
        if classifier_model is not None:
            # Example prediction on a new prompt
            new_prompt = "Hello, my name is Jane Smith and my social security is 901231-123.45. I support the democrat party."
            new_features = np.array(list(extract_features_for_ml(new_prompt).values())).reshape(1, -1)
            prediction = classifier_model.predict(new_features)[0]
            
            risk_levels = {4: 'C4', 3: 'C3', 2: 'C2', 1: 'C1', 0: 'C0'}
            print(f"\nPrediction for new prompt: '{new_prompt[:50]}...'")
            print(f"  Highest Predicted Risk Level: {risk_levels.get(prediction, 'Unknown')}")
            print(f"  Extracted Features: {extract_features_for_ml(new_prompt)}")


    # Outline the BERT-based approach
    outline_bert_classifier()