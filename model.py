import torch
import numpy as np
import pandas as pd
import emoji
import torch
from transformers import AutoModel, AutoTokenizer 

bertweet = AutoModel.from_pretrained("vinai/bertweet-base")
tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")

# INPUT TWEET IS ALREADY NORMALIZED!
# line = "SC has first two presumptive cases of coronavirus , DHEC confirms HTTPURL via @USER :cry:"

# input_ids = torch.tensor([tokenizer.encode(line)])

# with torch.no_grad():
#     features = bertweet(input_ids)  # Models outputs are now tuples

# print(features.last_hidden_state.shape, features.pooler_output.shape)

def load_data(filename: str):
    with open(filename, 'r') as file:
        lines = [line[:-1].split() for line in file]
    samples, start = [], 0
    for end, parts in enumerate(lines):
        if not parts:
            sample = [(token, tag.split('-')[-1]) for token, tag in lines[start:end]]
            samples.append(sample)
            start = end + 1
    if start < end:
        samples.append(lines[start:end])
    return samples
  
train_samples = load_data('data/train/train.txt')
val_samples = load_data('data/dev/dev.txt')
samples = train_samples + val_samples
schema = ['_'] + sorted({tag for sentence in samples for _, tag in sentence})

# print(len(schema))
# print(1)

