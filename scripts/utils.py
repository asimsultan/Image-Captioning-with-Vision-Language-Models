import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
from PIL import Image
import requests

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_data(examples, feature_extractor, tokenizer, max_length):
    pixel_values = feature_extractor(examples['image'], return_tensors='pt').pixel_values
    input_ids = tokenizer(examples['caption'], padding='max_length', truncation=True, max_length=max_length).input_ids
    input_ids = torch.tensor(input_ids)
    return {'pixel_values': pixel_values, 'input_ids': input_ids}

def collate_fn(batch):
    pixel_values = torch.stack([item['pixel_values'].squeeze(0) for item in batch])
    input_ids = torch.stack([item['input_ids'] for item in batch])
    return {'pixel_values': pixel_values, 'input_ids': input_ids}

def create_data_loader(dataset, batch_size, sampler):
    data_sampler = sampler(dataset)
    data_loader = DataLoader(dataset, sampler=data_sampler, batch_size=batch_size, collate_fn=collate_fn)
    return data_loader
