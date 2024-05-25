import torch
from transformers import BertForTokenClassification, BertTokenizer
from safetensors import safe_open
import os
from google.colab import drive


drive.mount('/content/drive', force_remount=True)

# The path of the model checkpoint directory on Google Drive
model_path = "/content/drive/MyDrive/bert_checkpoints"
latest_checkpoint = os.path.join(model_path, "checkpoint-12600")
print(f"Loading model from {latest_checkpoint}")

# Converting model.safetensors to pytorch_model.bin
safetensor_path = os.path.join(latest_checkpoint, "model.safetensors")
pytorch_model_path = os.path.join(latest_checkpoint, "pytorch_model.bin")

# Load weights from safetensor
with safe_open(safetensor_path, framework="pt") as f:
    state_dict = {key: f.get_tensor(key) for key in f.keys()}

# Initialize model and load state dict
model = BertForTokenClassification.from_pretrained('asafaya/bert-base-arabic', num_labels=len(label_mapping))
model.load_state_dict(state_dict)

# Save the model as pytorch_model.bin
torch.save(model.state_dict(), pytorch_model_path)
print(f"Model saved to {pytorch_model_path}")

# Load the model and tokenizer from the converted pytorch_model.bin
tokenizer = BertTokenizer.from_pretrained('asafaya/bert-base-arabic')
model = BertForTokenClassification.from_pretrained(latest_checkpoint)

# Prepare the Input Data
# An example of the input text
text = "ميشيل موناغان"

# Tokenize the input text
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

# Make Predictions
model.eval()
with torch.no_grad():
    outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits

predictions = torch.argmax(logits, dim=2)

# Decode the Predictions
# Define the label map 
id_to_label = {0: "O", 1: "B-PER", 2: "I-PER"}

predicted_labels = [id_to_label[label_id.item()] for label_id in predictions[0]]

# Print the tokens with their predicted labels
tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
for token, label in zip(tokens, predicted_labels):
    print(f"{token}: {label}")
