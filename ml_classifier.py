import json
import torch
from datasets import Dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    pipeline
)
from ml_setup import label_prompt

# ------------------------------
# 1. Load and prepare dataset
# ------------------------------
def load_labeled_data(file_path: str):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: {file_path} not found. Using small mock dataset.")
        # Use minimal mock data
        return [
            {"text": "Account IBAN BE68539007547034.", "label": 4},
            {"text": "Client CUST-1025, born 15/03/1985.", "label": 3},
            {"text": "Meeting with Microsoft in Paris.", "label": 2},
            {"text": "Final report for 2024 project.", "label": 1},
            {"text": "Draft new email for external communication.", "label": 0}
        ]

# Convert label from entity sensitivity
def assign_label(entities):
    levels = [e.get("sensitivity", "C0") for e in entities]
    if "C4" in levels: return 4
    if "C3" in levels: return 3
    if "C2" in levels: return 2
    if "C1" in levels: return 1
    return 0

def prepare_dataset(labeled_data):
    data_texts = []
    data_labels = []

    for item in labeled_data:
        text = item["text"]
        entities = item.get("entities", [])
        label = assign_label(entities)
        data_texts.append(text)
        data_labels.append(label)

    return Dataset.from_dict({"text": data_texts, "label": data_labels})

# ------------------------------
# 2. Initialize BERT
# ------------------------------
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=5)

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

# ------------------------------
# 3. Train BERT classifier
# ------------------------------
def train_bert_classifier(dataset):
    tokenized_dataset = dataset.map(tokenize, batched=True)
    tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = tokenized_dataset["train"]
    eval_dataset = tokenized_dataset["test"]

    training_args = TrainingArguments(
        output_dir="./bert_results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )

    trainer.train()
    return trainer

# ------------------------------
# 4. Prediction pipeline
# ------------------------------
def predict_risk(trainer, texts):
    classifier = pipeline("text-classification", model=trainer.model, tokenizer=tokenizer, return_all_scores=False)
    results = classifier(texts)
    risk_mapping = {0: "C0", 1: "C1", 2: "C2", 3: "C3", 4: "C4"}
    # Convert to risk level
    for i, r in enumerate(results):
        r["risk_level"] = risk_mapping[r["label"]]
    return results

# ------------------------------
# MAIN
# ------------------------------
if __name__ == "__main__":
    labeled_data = load_labeled_data("labeled_data/test.json")

    # If entities missing, use label_prompt to annotate
    for item in labeled_data:
        if "entities" not in item or not item["entities"]:
            res = label_prompt(item["text"], inject=False, use_spacy=True, use_c1_c2=False)
            item["entities"] = res["entities"]

    dataset = prepare_dataset(labeled_data)
    trainer = train_bert_classifier(dataset)

    # Example new prompt
    new_prompt = "Hello, my name is Jane Smith and my social security is 901231-123.45. I support the democrat party."

    # Predict risk
    result = predict_risk(trainer, [new_prompt])[0]
    print(f"\nPredicted risk level: {result['risk_level']} (label {result['label']})")

    # Print detected entities for context
    res = label_prompt(new_prompt, inject=False, use_spacy=True, use_c1_c2=False)
    print("\nDetected Entities:")
    for e in res['entities']:
        print(f"{e['entity']} ({e['type']}, {e['sensitivity']})")