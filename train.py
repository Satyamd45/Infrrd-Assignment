# Import necessary libraries
import pandas as pd
import os
from transformers import BertTokenizer, BertForTokenClassification, AdamW
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from tqdm import tqdm

# Set base paths - update with your directory structure
train_boxes_path = "/Users/satyamdoijode/Desktop/Infrrd/dataset/dataset/train/boxes_transcripts_labels"
val_boxes_path = "/Users/satyamdoijode/Desktop/Infrrd/dataset/dataset/val/boxes_transcripts"
val_ids_path = "/Users/satyamdoijode/Desktop/Infrrd/dataset/dataset/val/val_ids.tsv"
train_ids_path = "/Users/satyamdoijode/Desktop/Infrrd/dataset/dataset/train/train_ids.tsv"

# Load train and validation IDs
def load_ids(file_path):
    return pd.read_csv(file_path, sep='\t', header=None)[0].tolist()

train_ids = load_ids(train_ids_path)
val_ids = load_ids(val_ids_path)

# Custom Dataset Class
class TokenClassificationDataset(Dataset):
    def __init__(self, ids, boxes_path, tokenizer, max_len, labels_available=True):
        self.ids = ids
        self.boxes_path = boxes_path
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.labels_available = labels_available
        self.data = self.load_data()

    def load_data(self):
        data = []
        for doc_id in self.ids:
            tsv_path = os.path.join(self.boxes_path, f"{doc_id}.tsv")
            if os.path.exists(tsv_path):
                try:
                    df = pd.read_csv(tsv_path, sep='\t')
                    # Check if required columns are present
                    if 'transcript' in df.columns and (not self.labels_available or 'field' in df.columns):
                        data.append(df)
                    else:
                        print(f"Missing required columns in {tsv_path}. Columns found: {df.columns.tolist()}")
                except Exception as e:
                    print(f"Error loading {tsv_path}: {e}")
            else:
                print(f"File {tsv_path} not found")
        
        if not data:
            raise ValueError("No valid data found in the dataset. Please check the input files.")
        
        return pd.concat(data, ignore_index=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        encoding = self.tokenizer(
            row['transcript'],
            padding='max_length',
            max_length=self.max_len,
            truncation=True,
            return_tensors='pt'
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        
        if self.labels_available and 'field' in row:
            label = row['field']
            item['labels'] = torch.tensor(label, dtype=torch.long)
        else:
            item['labels'] = torch.tensor(-100)
        
        return item

# Initialize Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load Train and Validation Datasets
try:
    train_dataset = TokenClassificationDataset(train_ids, train_boxes_path, tokenizer, max_len=128, labels_available=True)
    val_dataset = TokenClassificationDataset(val_ids, val_boxes_path, tokenizer, max_len=128, labels_available=False)
except ValueError as e:
    print(e)
    exit()

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Model Definition
class TokenClassifier(nn.Module):
    def __init__(self, num_labels):
        super(TokenClassifier, self).__init__()
        self.bert = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return output

# Initialize Model
num_labels = len(train_dataset.data['field'].unique()) if 'field' in train_dataset.data else 0
if num_labels == 0:
    print("No labels found in the training dataset. Please check the input files.")
    exit()

model = TokenClassifier(num_labels)

# Optimizer
optimizer = AdamW(model.parameters(), lr=3e-5)

# Training Function
def train_model(model, train_loader, val_loader, optimizer, epochs=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask, labels=labels)
            loss = outputs.loss
            train_loss += loss.item()

            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}, Training loss: {train_loss/len(train_loader)}")
        validate_model(model, val_loader)

# Validation Function
def validate_model(model, val_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask, labels=labels)
            loss = outputs.loss
            val_loss += loss.item()

    print(f"Validation loss: {val_loss/len(val_loader)}")

# Train Model
train_model(model, train_loader, val_loader, optimizer, epochs=3)
