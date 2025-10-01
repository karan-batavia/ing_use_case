import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizer, BertModel
from torch.optim import AdamW
import numpy as np
from sklearn.preprocessing import StandardScaler

from ml_setup import label_prompt, RELIGION_KEYWORDS, ETHNICITY_KEYWORDS, SEXUAL_ORIENTATION_KEYWORDS, POLITICAL_KEYWORDS, HEALTH_KEYWORDS, load_prompts

# -------------------------
# Dataset Class
# -------------------------
class RiskDataset(Dataset):
    def __init__(self, texts, numeric_features, labels, tokenizer, max_len=128):
        self.texts = texts
        self.numeric_features = numeric_features
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        features = self.numeric_features[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'numeric_features': torch.tensor(features, dtype=torch.float),
            'label': torch.tensor(label, dtype=torch.long)
        }

# -------------------------
# Feature Extraction
# -------------------------
def extract_features_for_ml(text: str) -> list:
    result = label_prompt(text, inject=False, use_spacy=True, use_c1_c2=False)
    stats = result["statistics"]
    features = [
        stats.get("c4_count", 0),
        stats.get("c3_count", 0),
        stats.get("c2_count", 0),
        stats.get("c1_count", 0),
        stats.get("total_entities", 0),
        stats.get("spacy_detected", 0),
        len(text),
        len(text.split()),
        int("?" in text),
        int(any(k in text.lower() for k in RELIGION_KEYWORDS)),
        int(any(k in text.lower() for k in ETHNICITY_KEYWORDS)),
        int(any(k in text.lower() for k in SEXUAL_ORIENTATION_KEYWORDS)),
        int(any(k in text.lower() for k in POLITICAL_KEYWORDS)),
        int(any(k in text.lower() for k in HEALTH_KEYWORDS))
    ]
    return features

# -------------------------
# Hybrid BERT Classifier
# -------------------------
class BertHybridClassifier(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', numeric_dim=14, num_classes=5):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        bert_dim = self.bert.config.hidden_size
        self.fc1 = nn.Linear(bert_dim + numeric_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, input_ids, attention_mask, numeric_features):
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = bert_outputs.last_hidden_state[:,0,:]  # CLS token
        x = torch.cat([cls_embedding, numeric_features], dim=1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# -------------------------
# Prepare Dataset
# -------------------------
def prepare_dataset(data):
    texts = [d["text"] for d in data]
    features = np.array([extract_features_for_ml(t) for t in texts])
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Compute max risk level per sample using entity sensitivities
    labels = []
    for d in data:
        levels = [e.get("sensitivity", "C0") for e in d.get("entities", [])]
        if "C4" in levels: labels.append(4)
        elif "C3" in levels: labels.append(3)
        elif "C2" in levels: labels.append(2)
        elif "C1" in levels: labels.append(1)
        else: labels.append(0)
    
    return texts, features_scaled, np.array(labels), scaler

# -------------------------
# Training Loop with Early Stopping
# -------------------------
def train_model(model, train_loader, val_loader=None, epochs=5, lr=2e-5, device='cpu', patience=2):
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            numeric_features = batch['numeric_features'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids, attention_mask, numeric_features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}")
        
        # Validation
        if val_loader:
            model.eval()
            val_loss = 0
            all_preds, all_labels = [], []
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    numeric_features = batch['numeric_features'].to(device)
                    labels = batch['label'].to(device)
                    outputs = model(input_ids, attention_mask, numeric_features)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    preds = torch.argmax(outputs, dim=1).cpu().numpy()
                    all_preds.extend(preds)
                    all_labels.extend(labels.cpu().numpy())
            avg_val_loss = val_loss / len(val_loader)
            acc = np.mean(np.array(all_preds) == np.array(all_labels))
            print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {acc:.4f}")
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(model.state_dict(), "best_model.pt")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break
    # Load best model
    model.load_state_dict(torch.load("best_model.pt"))

# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    # Load raw prompts
    raw_texts = load_prompts("raw_prompts.txt")
    labeled_data = []
    for text in raw_texts:
        labeled_result = label_prompt(text, inject=False, use_spacy=True, use_c1_c2=False)
        labeled_data.append({
            "text": text,
            "entities": labeled_result["entities"]
        })
    
    # Prepare dataset
    texts, features, labels, scaler = prepare_dataset(labeled_data)
    
    # Split into train/val
    train_size = int(0.8 * len(texts))
    val_size = len(texts) - train_size
    
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    dataset = RiskDataset(texts, features, labels, tokenizer)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = BertHybridClassifier(numeric_dim=features.shape[1], num_classes=5)
    train_model(model, train_loader, val_loader, epochs=5, lr=2e-5, device=device, patience=2)
    
    # Predict on new prompt
    new_prompt = "Hello, my name is Jane Smith and my social security is 901231-123.45. I support the democrat party."
    new_features = torch.tensor(scaler.transform([extract_features_for_ml(new_prompt)]), dtype=torch.float).to(device)
    encoding = tokenizer.encode_plus(new_prompt, add_special_tokens=True, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
    model.eval()
    with torch.no_grad():
        logits = model(encoding['input_ids'].to(device), encoding['attention_mask'].to(device), new_features)
        pred_class = torch.argmax(logits, dim=1).item()
    print(f"\nPredicted risk level for new prompt: C{pred_class}")
    
    # Show entities and features
    result = label_prompt(new_prompt, inject=False, use_spacy=True, use_c1_c2=False)
    print("\nDetected Entities:")
    for e in result['entities']:
        print(f"{e['entity']} ({e['type']}, {e['sensitivity']})")
    
    print("\nExtracted Features:")
    features_dict = {
        "c4_count": result["statistics"].get("c4_count", 0),
        "c3_count": result["statistics"].get("c3_count", 0),
        "c2_count": result["statistics"].get("c2_count", 0),
        "c1_count": result["statistics"].get("c1_count", 0),
        "total_entities": result["statistics"].get("total_entities", 0),
        "spacy_ner_count": result["statistics"].get("spacy_detected", 0),
        "text_length": len(new_prompt),
        "word_count": len(new_prompt.split()),
        "has_question_mark": "?" in new_prompt,
        "has_religion_keyword": any(k in new_prompt.lower() for k in RELIGION_KEYWORDS),
        "has_ethnicity_keyword": any(k in new_prompt.lower() for k in ETHNICITY_KEYWORDS),
        "has_sexual_orientation_keyword": any(k in new_prompt.lower() for k in SEXUAL_ORIENTATION_KEYWORDS),
        "has_political_keyword": any(k in new_prompt.lower() for k in POLITICAL_KEYWORDS),
        "has_health_keyword": any(k in new_prompt.lower() for k in HEALTH_KEYWORDS)
    }
    for k, v in features_dict.items():
        print(f"{k}: {v}")