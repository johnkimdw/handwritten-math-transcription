import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import os, subprocess
from tqdm import tqdm
import random
import subprocess
import urllib.request
import tarfile
import re
import math
from torch.utils.data import Subset

from config import *
from dataset.hme_dataset import HMEDataset
from model import Encoder, Decoder, Seq2Seq, Attention
from utils import download_data, tokenize_latex, collate_variable_length_sequences
from dataset.hme_ink import read_inkml_file

# def create_model():
#     input_dim = 11                   
#     enc_hidden_dim = 256             
#     dec_hidden_dim = 256             
#     embed_dim = 128                  
#     output_dim = LATEX_VOCAB_SIZE    
#     encoder_num_layers = 2
#     decoder_num_layers = 2

#     encoder = Encoder(input_dim, enc_hidden_dim, num_layers=encoder_num_layers, bidirectional=True)
#     decoder = Decoder(output_dim, embed_dim, enc_hidden_dim, dec_hidden_dim, num_layers=decoder_num_layers)
#     model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)
#     return model

def train(model, train_loader, val_loader, epochs=100, lr=1e-3, clip=1.0, pad_idx=0):
    if not os.path.exists("model"):
        os.makedirs("model")

    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=2)
    crit  = nn.CrossEntropyLoss(ignore_index=pad_idx, label_smoothing=0.1)
    best  = float('inf')

    def tf_ratio(ep, k=5):
        return k/(k + math.exp(ep/ k))


    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch} [train]", leave=False)
        for src, L, trg in train_loader:
            src, L, trg = src.to(model.device), L.to(model.device), trg.to(model.device)
            opt.zero_grad()
            preds = model(src, L, trg, tf_ratio(epoch))
            B, T, V = preds.size()
            loss = crit(preds[:,1:].reshape(-1,V), trg[:,1:].reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            running_loss += loss.item()
            train_bar.set_postfix(train_loss=running_loss/(train_bar.n+1))
        avg_tr = running_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch} [valid]", leave=False)
        with torch.no_grad():
            for src, L, trg in val_loader:
                src, L, trg = src.to(model.device), L.to(model.device), trg.to(model.device)
                preds = model(src, L, trg, 0.0)
                B, T, V = preds.size()
                val_loss += crit(preds[:,1:].reshape(-1,V), trg[:,1:].reshape(-1)).item()
                val_bar.set_postfix(val_loss=val_loss/(val_bar.n+1))

        avg_v = val_loss/len(val_loader)
        sched.step(avg_v)

        if avg_v < best:
            best = avg_v
            torch.save(model.state_dict(), "model_best.pth")
        print(f"epoch {epoch} average train loss={avg_tr:.4f}  average val loss={avg_v:.4f}  tf={tf_ratio(epoch):.2f}")


        # for idx, batch in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
        #     # Get the inputs (data is a list of [inputs, labels])
            
        #     # inputs, latex_gt = data     # latex ground truth
        #     # inputs = inputs.to(DEVICE)
        #     # latex_gt = latex_gt.to(DEVICE)

        #     inputs, lengths, targets = batch
        #     inputs = inputs.to(DEVICE)
        #     lengths = lengths.to(DEVICE)
        #     targets = targets.to(DEVICE)
            
        #     # optimizer.zero_grad()
        #     # outputs = model(inputs)
        #     # loss = criterion(outputs, labels)
        #     # loss.backward()
        #     # optimizer.step()

        #     optimizer.zero_grad()
        #     output = model(inputs, lengths, targets, teacher_forcing_ratio=0.5)

        #     # loss = loss.detach().cpu().numpy()
        #     # inputs = inputs.detach().cpu().numpy()
        #     # labels = labels.detach().cpu().numpy()
        #     # running_loss += loss

        #     output_dim = output.shape[-1]
        #     output = output[:, 1:].reshape(-1, output_dim)
        #     targets = targets[:, 1:].reshape(-1)
        #     loss = criterion(output, targets)
        #     loss.backward()
        #     optimizer.step()
        #     running_loss += loss.item()
            
        # avg_train_loss = running_loss / len(train_loader)
        # print(f"epoch {epoch} training loss: {avg_train_loss:.4f}")

        # # evaluate the accuracy after each epoch
        # # acc = model.evaluate(model, val_loader, classes, device)
        # # if acc > best_acc:
        # #     print(f"Better validation accuracy achieved: {acc * 100:.2f}%")
        # #     best_acc = acc
        # #     print(f"Saving this model as: {my_best_model}")
        # #     torch.save(model.state_dict(), my_best_model)

        # model.eval()
        # val_loss = 0.0
        # with torch.no_grad():
        #     for batch in val_loader:
        #         inputs, lengths, targets = batch
        #         inputs = inputs.to(DEVICE)
        #         lengths = lengths.to(DEVICE)
        #         targets = targets.to(DEVICE)
                
        #         output = model(inputs, lengths, targets, teacher_forcing_ratio=0.0)  # no teacher forcing during evaluation
        #         output_dim = output.shape[-1]
        #         output = output[:, 1:].reshape(-1, output_dim)
        #         targets_flat = targets[:, 1:].reshape(-1)
        #         loss = criterion(output, targets_flat)
        #         val_loss += loss.item()
        
        # avg_val_loss = val_loss / len(val_loader)
        # print(f"epoch {epoch} validation loss: {avg_val_loss:.4f}")
        
        # # save model
        # if avg_val_loss < best_val_loss:
        #     best_val_loss = avg_val_loss
        #     torch.save(model.state_dict(), f"model/model_best_{epoch}.pth")
        #     print(f"model saved as 'model/model_best_{epoch}.pth'.")


def test_model(model, test_loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    # examples to check 
    examples = []

    EOS = LATEX_VOCAB['<eos>']
    
    with torch.no_grad():
        for src, lengths, trg in test_loader:
            # move to device
            src, lengths, trg = src.to(DEVICE), lengths.to(DEVICE), trg.to(DEVICE)

            # forward with no teacher forcing
            outputs = model(src, lengths, trg, tf_ratio=0.0)  
            B, T, V = outputs.size()

            # compute loss (skip the first SOS step)
            loss = criterion(
                outputs[:, 1:].reshape(-1, V),
                trg[:,    1:].reshape(-1)
            )
            total_loss += loss.item()

            # greedy predictions
            preds = outputs.argmax(dim=-1)  # (B, T)

            # compare each sequence
            for i in range(B):
                pred_seq = preds[i, 1:].tolist()
                true_seq =  trg[i, 1:].tolist()

                # truncate at EOS if present
                if EOS in pred_seq:
                    pred_seq = pred_seq[:pred_seq.index(EOS)]
                if EOS in true_seq:
                    true_seq = true_seq[:true_seq.index(EOS)]

                # exact-match check
                if pred_seq == true_seq:
                    correct += 1

                # collect up to 5 examples
                if len(examples) < 5:
                    pred_ltx = indices_to_latex(pred_seq, LATEX_VOCAB_REVERSE)
                    true_ltx = indices_to_latex(true_seq, LATEX_VOCAB_REVERSE)
                    examples.append((pred_ltx, true_ltx))

                total += 1

    avg_loss = total_loss / len(test_loader)
    exact_match_acc = correct / total if total > 0 else 0.0

    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Accuracy:  {exact_match_acc:.4f} ({correct}/{total})\n")
    print("Some sample predictions:")
    for pred, true in examples:
        print(f"Predicted: {pred}")
        print(f"Actual:    {true}")

    return avg_loss, exact_match_acc, examples



def inference(model, ink_file_path=None, ink_object=None, max_length=150):
    """
    Ink file/object --> model --> LaTeX string
    """
    model.eval()
    
    # get the ink object first
    if ink_object is None and ink_file_path is not None: ink = read_inkml_file(ink_file_path)
    elif ink_object is not None:  ink = ink_object
    else: raise ValueError("Either ink_file_path or ink_object must be provided")
    
    # ground_truth_latex = ink_object.annotations.get('normalizedLabel')

    
    # extract features from ink object
    dataset = HMEDataset(root_dir="mathwriting-2024", split="train")
    ink_features = dataset.extract_features(ink)
    ground_truth_latex = ink.annotations['normalizedLabel']
    
    # handle empty feature case
    if ink_features.size(0) == 0: return "<empty_features>"
    
    # add batch dimension (1, seq_len, feature_dim)
    input_tensor = ink_features.unsqueeze(0).to(DEVICE)
    input_length = torch.tensor([ink_features.size(0)]).to(DEVICE)
    
    with torch.no_grad():
        # initialize with SOS token
        input_token = torch.tensor([LATEX_VOCAB['<sos>']]).to(DEVICE)
        
        # encoder forward pass
        encoder_outputs, (hidden, cell) = model.encoder(input_tensor, input_length)
        
        # handle bidirectional encoder
        if model.encoder.bidirectional:
            hidden = hidden.view(model.encoder.num_layers, 2, hidden.size(1), hidden.size(2)).sum(dim=1)
            cell = cell.view(model.encoder.num_layers, 2, cell.size(1), cell.size(2)).sum(dim=1)
        
        # attention mask
        mask = model.create_mask(input_tensor)
        
        # generate sequence
        output_indices = [LATEX_VOCAB['<sos>']]
        attention_weights = []
        
        for _ in range(max_length):
            # forward pass through decoder
            prediction, hidden, cell, attn_weights = model.decoder(
                input_token, hidden, cell, encoder_outputs, mask
            )
            
            # store attention weights if needed
            attention_weights.append(attn_weights.squeeze(0).cpu())
            
            # get most likely token
            top_token = prediction.argmax(1).item()
            output_indices.append(top_token)
            
            # end if EOS token
            if top_token == LATEX_VOCAB['<eos>']:break
            
            # update input token for next step
            input_token = torch.tensor([top_token]).to(DEVICE)
    
    # convert indices to LaTeX
    latex_output = indices_to_latex(output_indices, LATEX_VOCAB_REVERSE)
    
    return latex_output, ground_truth_latex, attention_weights



def main():
    print(DEVICE)
    
    # # create model
    # model = create_model()
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=LATEX_PAD_TOKEN, label_smoothing=0.1)  

    root_dir = download_data(url="https://storage.googleapis.com/mathwriting_data/mathwriting-2024.tgz")
    print(root_dir)
    # init dataset

    full_train_dataset   = HMEDataset(root_dir, "train")
    valid_dataset   = HMEDataset(root_dir, "valid")
    test_dataset    = HMEDataset(root_dir, "test")

    print(f"Found {len(full_train_dataset.ink_files)} files in {full_train_dataset.split} split")
    print(f"Found {len(valid_dataset.ink_files)} files in {valid_dataset.split} split")
    print(f"Found {len(test_dataset.ink_files)} files in {test_dataset.split} split")

    SUBSET_SIZE = 50000
    all_idx     = list(range(len(full_train_dataset)))
    random.shuffle(all_idx)
    subset_idx  = all_idx[:SUBSET_SIZE]
    train_dataset    = Subset(full_train_dataset, subset_idx)
    
    # print(train_dataset[0])
    

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, collate_fn=collate_variable_length_sequences)
    valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, collate_fn=collate_variable_length_sequences)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, collate_fn=collate_variable_length_sequences)
    
    # print(train_dataloader)
    # print(next(iter(train_dataloader)))

    sample_feat, _ = train_dataset[0]           
    input_dim = sample_feat.shape[1]
    
    encoder = Encoder(input_dim=input_dim)
    decoder = Decoder(
        output_dim=len(LATEX_VOCAB), 
        embed_dim=64, encoder_hidden_dim=128, decoder_hidden_dim=128
    )
    model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)
    
    
    # inspect one batch
    # for batch in train_dataloader:
    #     features, lengths, labels = batch
    #     print(lengths)
    
    # train(model, train_dataloader, valid_dataloader, EPOCHS, optimizer, criterion)
    
    model_path = "model/model_best.pth"
    if os.path.exists(model_path):
        print(f"Loading pre-trained model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    else:
        print("Training new model")
        train(model, train_dataloader, valid_dataloader, epochs=EPOCHS, lr=1e-3)
    
    
    ink_path = "mathwriting-2024/test/0a0b310001bedb73.inkml"
    print(f"ink_path: {ink_path}")
    predicted_latex, actual_latex, attention = inference(model, ink_file_path=ink_path)
    print(f"Prediction: {predicted_latex}")
    print(f"Actual: {actual_latex}")

    print("Running full evaluation on TEST split:")
    test_loss, test_acc, examples = test_model(model, test_dataloader, criterion)

    


def indices_to_latex(indices, vocab_reverse):
    """Convert a sequence of token indices back to LaTeX string"""
    tokens = [vocab_reverse[idx] for idx in indices if idx in vocab_reverse and idx not in [LATEX_VOCAB['<pad>'], LATEX_VOCAB['<sos>'], LATEX_VOCAB['<eos>']]]
    return ''.join(tokens)


if __name__ == "__main__":
    main()
    
