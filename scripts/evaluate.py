import torch, os
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
from datasets import load_dataset
from utils import get_device, preprocess_data, create_data_loader

# Parameters
model_dir = './models'
max_length = 16
batch_size = 4

# Load Model and Tokenizer
model = VisionEncoderDecoderModel.from_pretrained(model_dir)
feature_extractor = ViTFeatureExtractor.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# Device
device = get_device()
model.to(device)

# Load Dataset
val_dataset = load_dataset('coco', '2017', split='validation[:1%]')

# Preprocess Data
val_dataset = val_dataset.map(lambda x: preprocess_data(x, feature_extractor, tokenizer, max_length), batched=True)

# DataLoader
val_loader = create_data_loader(val_dataset, batch_size, SequentialSampler)

# Evaluation Function
def evaluate(model, data_loader, device, tokenizer):
    model.eval()
    predictions = []
    references = []

    with torch.no_grad():
        for batch in data_loader:
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)

            outputs = model.generate(pixel_values=pixel_values, max_length=max_length)
            decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(input_ids, skip_special_tokens=True)

            predictions.extend(decoded_preds)
            references.extend(decoded_labels)

    return predictions, references

# Evaluate
predictions, references = evaluate(model, val_loader, device, tokenizer)
print("Predictions:", predictions[:5])
print("References:", references[:5])

# Save the predictions and references
output_dir = './outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

with open(os.path.join(output_dir, 'predictions.txt'), 'w') as f:
    for pred in predictions:
        f.write(pred + '\n')

with open(os.path.join(output_dir, 'references.txt'), 'w') as f:
    for ref in references:
        f.write(ref + '\n')
