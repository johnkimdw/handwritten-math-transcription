import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.nn.functional as F

import os, subprocess
import tqdm
import random
import subprocess
import urllib.request
import tarfile
import re

from config import *
from dataset.hme_dataset import HMEDataset
from model import Encoder, Decoder, Seq2Seq, Attention
from utils import download_data, tokenize_latex, collate_variable_length_sequences

def create_model():
    input_dim = 11                   
    enc_hidden_dim = 256             
    dec_hidden_dim = 256             
    embed_dim = 128                  
    output_dim = LATEX_VOCAB_SIZE    
    encoder_num_layers = 2
    decoder_num_layers = 2

    encoder = Encoder(input_dim, enc_hidden_dim, num_layers=encoder_num_layers, bidirectional=True)
    decoder = Decoder(output_dim, embed_dim, enc_hidden_dim, dec_hidden_dim, num_layers=decoder_num_layers)
    model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)
    return model

def train(model, train_loader, val_loader, epochs, optimizer, criterion):
    if not os.path.exists("model"):
        os.makedirs("model")

    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        print(f"starting epoch {epoch}:")
        for idx, batch in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
            # Get the inputs (data is a list of [inputs, labels])
            
            # inputs, latex_gt = data     # latex ground truth
            # inputs = inputs.to(DEVICE)
            # latex_gt = latex_gt.to(DEVICE)

            inputs, lengths, targets = batch
            inputs = inputs.to(DEVICE)
            lengths = lengths.to(DEVICE)
            targets = targets.to(DEVICE)
            
            # optimizer.zero_grad()
            # outputs = model(inputs)
            # loss = criterion(outputs, labels)
            # loss.backward()
            # optimizer.step()

            optimizer.zero_grad()
            output = model(inputs, lengths, targets, teacher_forcing_ratio=0.5)

            # loss = loss.detach().cpu().numpy()
            # inputs = inputs.detach().cpu().numpy()
            # labels = labels.detach().cpu().numpy()
            # running_loss += loss

            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            targets = targets[:, 1:].reshape(-1)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        avg_train_loss = running_loss / len(train_loader)
        print(f"epoch {epoch} training loss: {avg_train_loss:.4f}")

        # evaluate the accuracy after each epoch
        # acc = model.evaluate(model, val_loader, classes, device)
        # if acc > best_acc:
        #     print(f"Better validation accuracy achieved: {acc * 100:.2f}%")
        #     best_acc = acc
        #     print(f"Saving this model as: {my_best_model}")
        #     torch.save(model.state_dict(), my_best_model)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs, lengths, targets = batch
                inputs = inputs.to(DEVICE)
                lengths = lengths.to(DEVICE)
                targets = targets.to(DEVICE)
                
                output = model(inputs, lengths, targets, teacher_forcing_ratio=0.0)  # no teacher forcing during evaluation
                output_dim = output.shape[-1]
                output = output[:, 1:].reshape(-1, output_dim)
                targets_flat = targets[:, 1:].reshape(-1)
                loss = criterion(output, targets_flat)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"epoch {epoch} validation loss: {avg_val_loss:.4f}")
        
        # save model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"model/model_best_{epoch}.pth")
            print(f"model saved as 'model/model_best_{epoch}.pth'.")

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # create model
    model = create_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=LATEX_PAD_TOKEN)  

    root_dir = download_data()
    print(root_dir)
    # init dataset

    train_dataset   = HMEDataset(root_dir, "train")
    valid_dataset   = HMEDataset(root_dir, "valid")
    test_dataset    = HMEDataset(root_dir, "test")
    
    print(f"Found {len(train_dataset.ink_files)} files in {train_dataset.split} split")
    print(f"Found {len(valid_dataset.ink_files)} files in {valid_dataset.split} split")
    print(f"Found {len(test_dataset.ink_files)} files in {test_dataset.split} split")
    
    print(train_dataset[0])
    

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, collate_fn=collate_variable_length_sequences)
    valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, collate_fn=collate_variable_length_sequences)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, collate_fn=collate_variable_length_sequences)
    
    print(train_dataloader)
    # print(train_dataloader[0])
    
    # inspect one batch
    for batch in train_dataloader:
        features, lengths, labels = batch
        print(lengths)
    
    train(model, train_dataloader, valid_dataloader, EPOCHS, optimizer, criterion)
    
    model_path = "model/model_best.pth"
    if os.path.exists(model_path):
        print(f"Loading pre-trained model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("Training new model")
        train(model, train_dataloader, valid_dataloader, EPOCHS, optimizer, criterion)
    



def test_model(model, test_loader, criterion):
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    
    # Store some examples for qualitative analysis
    examples = []
    
    with torch.no_grad():
        for batch in tqdm.tqdm(test_loader, desc="Testing"):
            inputs, lengths, targets = batch
            inputs = inputs.to(DEVICE)
            lengths = lengths.to(DEVICE)
            targets = targets.to(DEVICE)
            
            # Forward pass without teacher forcing
            outputs = model(inputs, lengths, targets, teacher_forcing_ratio=0.0)
            
            # Calculate loss
            output_dim = outputs.shape[-1]
            outputs_flat = outputs[:, 1:].reshape(-1, output_dim)
            targets_flat = targets[:, 1:].reshape(-1)
            loss = criterion(outputs_flat, targets_flat)
            total_loss += loss.item()
            
            # Calculate accuracy
            predictions = outputs.argmax(dim=-1)
            for i in range(len(targets)):
                pred_seq = predictions[i, 1:].cpu().numpy()  # Skip SOS token
                true_seq = targets[i, 1:].cpu().numpy()
                
                # Find index of first EOS token or end of sequence
                pred_end = np.argwhere(pred_seq == LATEX_VOCAB['<eos>']).flatten()
                pred_end = pred_end[0] if len(pred_end) > 0 else len(pred_seq)
                
                true_end = np.argwhere(true_seq == LATEX_VOCAB['<eos>']).flatten()
                true_end = true_end[0] if len(true_end) > 0 else len(true_seq)
                
                pred_seq = pred_seq[:pred_end]
                true_seq = true_seq[:true_end]
                
                # Check for exact sequence match
                if len(pred_seq) == len(true_seq) and np.all(pred_seq == true_seq):
                    correct_predictions += 1
                
                total_predictions += 1
                
                # Store some examples for qualitative analysis
                if len(examples) < 5:
                    pred_latex = indices_to_latex(pred_seq, LATEX_VOCAB_REVERSE)
                    true_latex = indices_to_latex(true_seq, LATEX_VOCAB_REVERSE)
                    examples.append((pred_latex, true_latex))
    
    avg_loss = total_loss / len(test_loader)
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    
    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Exact Match Accuracy: {accuracy:.4f} ({correct_predictions}/{total_predictions})")
    
    print("\nSample predictions:")
    for pred, true in examples:
        print(f"Predicted: {pred}")
        print(f"True: {true}")
        print("-" * 50)
    
    return avg_loss, accuracy, examples


if __name__ == "__main__":
    main()
    