import numpy as np
from tqdm import tqdm
import torch
import torch
from transformers import AutoModel, AutoTokenizer, BertweetTokenizer


# load model and tokenizer
bertweet = AutoModel.from_pretrained("vinai/bertweet-base")
tokenizer = BertweetTokenizer.from_pretrained("vinai/bertweet-base", normalization=True)

#load data
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
  
# load data
train_samples = load_data('data/train/train.txt')
val_samples = load_data('data/dev/dev.txt')
samples = train_samples + val_samples
schema = ['_'] + sorted({tag for sentence in samples for _, tag in sentence})

def tokenize_sample(sample):
    seq = [
               (subtoken, tag)
               for token, tag in sample
               for subtoken in tokenizer(token)['input_ids'][1:-1]
           ]
    return [(3, 'O')] + seq + [(4, 'O')]

def preprocess(samples):
    tag_index = {tag: i for i, tag in enumerate(schema)}
    tokenized_samples = list(tqdm(map(tokenize_sample, samples)))
    max_len = max(map(len, tokenized_samples))
    X = np.zeros((len(samples), max_len), dtype=np.int32)
    y = np.zeros((len(samples), max_len), dtype=np.int32)
    for i, sentence in enumerate(tokenized_samples):
        for j, (subtoken_id, tag) in enumerate(sentence):
            X[i, j] = subtoken_id
            y[i,j] = tag_index[tag]
    return X, y

X_train, y_train = preprocess(train_samples)
X_val, y_val = preprocess(val_samples)