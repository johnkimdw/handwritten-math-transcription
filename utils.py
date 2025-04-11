from torch.nn.utils.rnn import pad_sequence
import torch

import os, subprocess
import subprocess
import urllib.request
import tarfile
import re

from config import *


def download_data(url="https://storage.googleapis.com/mathwriting_data/mathwriting-2024-excerpt.tgz"):
    filename = url.split("/")[-1]
    dirname = filename.split('.')[0]

    if not os.path.exists(dirname):
        if not os.path.exists(filename):
            print(f"downloading {filename}...")
            urllib.request.urlretrieve(url, filename)
        else:
            print(f"{filename} already downloaded.")

        print(f"extracting {filename}...")
        with tarfile.open(filename, "r:gz") as tar:
            tar.extractall()
        
        os.remove(filename)
        print(f"extraction complete. removed {filename}.")
    else:
        print(f"{dirname} already exists. skipping download.")


    return dirname

def tokenize_latex(latex_str, vocab):
    tokens = [vocab['<sos>']]
    
    pattern = re.compile(r'(\\[a-zA-Z]+)|(\d+)|(\S)')
    for match in pattern.finditer(latex_str):
        token = match.group(0)
        if token in vocab:
            tokens.append(vocab[token])
        else:
            for char in token:
                if char in vocab:
                    tokens.append(vocab[char])
    tokens.append(vocab['<eos>'])

    return tokens

def collate_variable_length_sequences(batch):
    feature_vectors, labels = zip(*batch)
    # print(len(feature_vectors), len(labels))
    
    # padded_features will have shape: [batch_size, max_seq_len, feature_dim]
    padded_features = pad_sequence(feature_vectors, batch_first=True, padding_value=0.0)
    lengths = torch.tensor([vec.size(0) for vec in feature_vectors]) # original lengths
    
    tokenized_labels = [torch.tensor(tokenize_latex(label, LATEX_VOCAB), dtype=torch.long) for label in labels]
    padded_labels = pad_sequence(tokenized_labels, batch_first=True, padding_value=LATEX_VOCAB['<pad>'])

    return padded_features, lengths, padded_labels
