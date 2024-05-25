import re
from collections import defaultdict
import zipfile

def extract_labels_from_zip(zip_file_path, text_file_name, encoding='utf-8-sig'):
    label_count = defaultdict(int)
    try:
        with zipfile.ZipFile(zip_file_path, 'r') as z:
            with z.open(text_file_name, 'r') as file:
                for line in file:
                    line = line.decode(encoding).strip()
                    labels = re.findall(r'<([A-Za-z_]+)>', line)
                    for label in labels:
                        label_count[label] += 1
    except Exception as e:
        print(f"Error reading data: {e}")
        return {}
    return label_count

# unzipping dataset
zip_file_path = '/content/drive/My Drive/WIKIFANE_gazet copy.txt.zip'
text_file_name = 'WIKIFANE_gazet copy.txt'
labels = extract_labels_from_zip(zip_file_path, text_file_name)

# Print all unique tags and their counts
if not labels:  
    print("No labels found.")
else:
    print(f"Total unique tags: {len(labels)}")
    for label, count in labels.items():
        print(f"{label}: {count}")

# Save unique tags to a file
with open('unique_tags.txt', 'w', encoding='utf-8') as f:
    for label in labels.keys():
        f.write(label + '\n')














import json

# Load unique tags from the file
with open('unique_tags.txt', 'r', encoding='utf-8') as f:
    unique_tags = [line.strip() for line in f]

# Create label_mapping
label_mapping = {tag: idx for idx, tag in enumerate(unique_tags)}

# Save label_mapping to a JSON file
with open('label_mapping.json', 'w', encoding='utf-8') as f:
    json.dump(label_mapping, f, ensure_ascii=False, indent=4)

# Print all unique tags and their mappings
print(f"Total tags mapped: {len(label_mapping)}")
for tag, idx in label_mapping.items():
    print(f"{tag}: {idx}")













# Install required libraries
"""!pip install transformers[torch]
!pip install accelerate -U"""

from transformers import BertTokenizer, BertForTokenClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
import torch
import os
import zipfile
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import logging
import numpy as np
import re
import json
from google.colab import drive
import time

# Mount Google Drive
drive.mount('/content/drive')

# Copy data to local directory for faster access
!cp "/content/drive/My Drive/WIKIFANE_gazet copy.txt.zip" /content/

# Load the comprehensive label_mapping
with open('label_mapping.json', 'r', encoding='utf-8') as f:
    label_mapping = json.load(f)

# Test CUDA setup
try:
    x = torch.tensor([1., 2.], device='cuda')
    print(f"CUDA setup works: {x}")
except RuntimeError as e:
    print(f"CUDA error: {e}")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_split_data(zip_file_path, text_file_name, test_size=0.1, random_state=42, encoding='utf-8-sig'):
    texts = []
    labels = []
    try:
        with zipfile.ZipFile(zip_file_path, 'r') as z:
            with z.open(text_file_name, 'r') as file:
                for line in file:
                    line = line.decode(encoding).strip()
                    parts = re.split(r'(<[^>]+>)', line)  # Split data by tags 
                    text = ""
                    line_labels = []
                    for part in parts:
                        if re.match(r'<[^>]+>', part):  
                            label = re.sub(r'[<>/]', '', part)  # Remove <, >, /
                            if label in label_mapping:  # Only keep tags in label_mapping
                                line_labels.append(label)
                        else:
                            text += part.strip() + " "
                    text = text.strip()
                    if text and line_labels:  # Only append if text is not empty
                        texts.append(text)
                        labels.append(line_labels[0])  # Use the first label found
        logging.info(f'Successfully read data from {text_file_name}')
    except Exception as e:
        logging.error(f'Error reading data: {e}')
        return [], [], [], []

    try:
        texts_train, texts_eval, labels_train, labels_eval = train_test_split(
            texts, labels, test_size=test_size, random_state=random_state
        )
        logging.info(f'Data split into training and evaluation sets with test_size={test_size}')
    except Exception as e:
        logging.error(f'Error splitting data: {e}')
        return [], [], [], []

    return texts_train, texts_eval, labels_train, labels_eval

class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.tokenizer = tokenizer
        self.texts = texts
        self.labels = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label_id = self.labels[idx]
        encoding = self.tokenizer(
            text, return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_len
        )
        label_ids = torch.tensor([label_id] * encoding['input_ids'].size(1), dtype=torch.long)

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': label_ids
        }

def convert_labels_to_ids(label):
    return label_mapping.get(label, -1)  # Default to -1 if label is not found

# Measure data loading and splitting time
start_time = time.time()
train_texts, eval_texts, train_labels, eval_labels = load_and_split_data(
    '/content/WIKIFANE_gazet copy.txt.zip', 'WIKIFANE_gazet copy.txt'
)
end_time = time.time()
print(f"Data loading and splitting took {end_time - start_time} seconds")

# Debugging step: Print all unique original labels that are not mapped
unmapped_labels = set()
for label in train_labels:
    if convert_labels_to_ids(label) == -1:
        unmapped_labels.add(label)

for label in eval_labels:
    if convert_labels_to_ids(label) == -1:
        unmapped_labels.add(label)

print("Unmapped labels:")
for label in unmapped_labels:
    print(label)

# map labels to their IDs
train_labels = [convert_labels_to_ids(label) for label in train_labels]
eval_labels = [convert_labels_to_ids(label) for label in eval_labels]

# Filter out any data points with unmapped labels (-1)
filtered_train_texts = [text for text, label in zip(train_texts, train_labels) if label != -1]
filtered_train_labels = [label for label in train_labels if label != -1]
filtered_eval_texts = [text for text, label in zip(eval_texts, eval_labels) if label != -1]
filtered_eval_labels = [label for label in eval_labels if label != -1]

# Debug: Check sizes of original and filtered datasets
print(f"Original training set size: {len(train_texts)}")
print(f"Original evaluation set size: {len(eval_texts)}")
print(f"Filtered training set size: {len(filtered_train_texts)}")
print(f"Filtered evaluation set size: {len(filtered_eval_texts)}")

# Ensure lengths of filtered lists are consistent
assert len(filtered_train_texts) == len(filtered_train_labels), "Mismatched train texts and labels."
assert len(filtered_eval_texts) == len(filtered_eval_labels), "Mismatched eval texts and labels."

# Ensure labels are within range
num_labels = len(label_mapping)
for label_id in filtered_train_labels + filtered_eval_labels:
    if label_id < 0 or label_id >= num_labels:
        raise ValueError(f"Label ID {label_id} out of range.")

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained('asafaya/bert-base-arabic')

# Initialize the datasets
train_dataset = CustomDataset(filtered_train_texts, filtered_train_labels, tokenizer)
eval_dataset = CustomDataset(filtered_eval_texts, filtered_eval_labels, tokenizer)

def data_collator(features):
    batch = {
        'input_ids': torch.stack([f['input_ids'] for f in features]),
        'attention_mask': torch.stack([f['attention_mask'] for f in features]),
        'labels': torch.stack([f['labels'] for f in features]),
    }
    return batch

# Setup training arguments
output_dir = '/content/drive/MyDrive/bert_checkpoints'
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=200,  
    save_strategy="steps",
    save_steps=200,  
    load_best_model_at_end=True
)

# Load model with the correct number of labels
model = BertForTokenClassification.from_pretrained('asafaya/bert-base-arabic', num_labels=num_labels)

# custom compute_metrics function defined
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    true_labels = labels.flatten()
    pred_labels = predictions.flatten()

    report = classification_report(true_labels, pred_labels, zero_division=0)
    print(report)
    return {"accuracy": (true_labels == pred_labels).mean()}

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# Inspect data before training
for i in range(min(3, len(train_dataset))):
    sample = train_dataset[i]
    print(f"Sample {i}:")
    print(f"Input IDs: {sample['input_ids']}")
    print(f"Attention Mask: {sample['attention_mask']}")
    print(f"Labels: {sample['labels']}")

"""Start training, to resume training from the last checkpoint, check if there are any checkpoints"""
checkpoint_dir = training_args.output_dir

# Find the last checkpoint
last_checkpoint = None
if os.path.isdir(checkpoint_dir):
    checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint-")]
    if checkpoints:
        checkpoints.sort(key=lambda x: int(x.split("-")[1]), reverse=True)
        last_checkpoint = os.path.join(checkpoint_dir, checkpoints[0])

if last_checkpoint:
    print(f"Resuming training from checkpoint: {last_checkpoint}")
    trainer.train(resume_from_checkpoint=last_checkpoint)
else:
    print("No checkpoint found. Starting from scratch.")
    trainer.train()
