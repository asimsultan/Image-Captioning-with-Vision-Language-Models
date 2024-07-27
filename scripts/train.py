import os
import torch
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer, AdamW, get_scheduler
from datasets import load_dataset
from utils import get_device, preprocess_data, create_data_loader

# Parameters
model_name = 'nlpconnect/vit-gpt2-image-captioning'
max_length = 16
batch_size = 4
epochs = 3
learning_rate = 5e-5

# Load Dataset
dataset = load_dataset('coco', '2017', split='train[:1%]')
val_dataset = load_dataset('coco', '2017', split='validation[:1%]')

# Feature Extractor and Tokenizer
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Preprocess Data
train_dataset = dataset.map(lambda x: preprocess_data(x, feature_extractor, tokenizer, max_length), batched=True)
val_dataset = val_dataset.map(lambda x: preprocess_data(x, feature_extractor, tokenizer, max_length), batched=True)

# DataLoader
train_loader = create_data_loader(train_dataset, batch_size, RandomSampler)
val_loader = create_data_loader(val_dataset, batch_size, SequentialSampler)

# Model
device = get_device()
model = VisionEncoderDecoderModel.from_pretrained(model_name)
model.to(device)

# Optimizer and Scheduler
optimizer = AdamW(model.parameters(), lr=learning_rate)
num_training_steps = len(train_loader) * epochs
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

# Training Function
def train_epoch(model, data_loader, optimizer, device, scheduler):
    model.train()
    total_loss = 0

    for batch in data_loader:
        pixel_values = batch['pixel_values'].to(device)
        input_ids = batch['input_ids'].to(device)

        outputs = model(pixel_values=pixel_values, labels=input_ids)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    avg_loss = total_loss / len(data_loader)
    return avg_loss

# Training Loop
for epoch in range(epochs):
    train_loss = train_epoch(model, train_loader, optimizer, device, lr_scheduler)
    print(f'Epoch {epoch+1}/{epochs}')
    print(f'Train Loss: {train_loss}')

# Save Model
model_dir = './models'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)
