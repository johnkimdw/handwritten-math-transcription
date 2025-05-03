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
from model import Encoder, Decoder, Seq2Seq
from utils import download_data, tokenize_latex, collate_variable_length_sequences
from dataset.hme_ink import read_inkml_file
from corrector import correct_latex

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

def train(model, train_loader, val_loader, epochs=50, lr=1e-3, clip=1.0, pad_idx=0):
    if not os.path.exists("model_att"):
        os.makedirs("model_att")

    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=2)
    crit  = nn.CrossEntropyLoss(ignore_index=pad_idx, label_smoothing=0.1)
    best  = float('inf')

    def tf_ratio(ep, k=5):
        return k/(k + math.exp(ep/ k))


    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        # train_bar = tqdm(train_loader, desc=f"Epoch {epoch} [train]", leave=False)
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
            # train_bar.set_postfix(train_loss=running_loss/(train_bar.n+1))
        avg_tr = running_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        # val_bar = tqdm(val_loader, desc=f"Epoch {epoch} [valid]", leave=False)
        with torch.no_grad():
            for src, L, trg in val_loader:
                src, L, trg = src.to(model.device), L.to(model.device), trg.to(model.device)
                preds = model(src, L, trg, 0.0)
                B, T, V = preds.size()
                val_loss += crit(preds[:,1:].reshape(-1,V), trg[:,1:].reshape(-1)).item()
                # val_bar.set_postfix(val_loss=val_loss/(val_bar.n+1))

        avg_v = val_loss/len(val_loader)
        sched.step(avg_v)

        if avg_v < best:
            best = avg_v
            torch.save(model.state_dict(), f"model_v3_{epoch}.pth")
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
    examples = []
    EOS = LATEX_VOCAB['<eos>']
    
    correct_orig = 0
    correct_fixed = 0

    for src, lengths, trg in tqdm(test_loader, desc="testing", leave=False):
        src, lengths, trg = src.to(DEVICE), lengths.to(DEVICE), trg.to(DEVICE)

        # no teacher forcing
        outputs = model(src, lengths, trg, teacher_forcing_ratio=0.0)
        B, T, V = outputs.size()

        # compute and accumulate loss
        loss = criterion(
            outputs[:, 1:].reshape(-1, V),
            trg[:,    1:].reshape(-1)
        )
        total_loss += loss.item()

        # greedy decode and accuracy counting
        preds = outputs.argmax(dim=-1)
        for i in range(B):
            pseq = preds[i,1:].tolist()
            tseq = trg[i,1:].tolist()
            if EOS in pseq: pseq = pseq[:pseq.index(EOS)]
            if EOS in tseq: tseq = tseq[:tseq.index(EOS)]
            
            orig_latex = indices_to_latex(pseq, LATEX_VOCAB_REVERSE)
            true_latex = indices_to_latex(tseq, LATEX_VOCAB_REVERSE)
            
            # correct latex
            fixed_latex = correct_latex(orig_latex)
            
            if orig_latex == true_latex: correct_orig += 1
            if fixed_latex == true_latex: correct_fixed += 1
                
            if fixed_latex == true_latex: correct += 1
                
            if len(examples) < 5:
                examples.append((
                    orig_latex,
                    fixed_latex,
                    true_latex
                ))
            total += 1
           

    avg_loss = total_loss / len(test_loader)
    exact_match_acc = correct / total if total else 0.0

    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Exact Match Accuracy: {exact_match_acc:.4f} ({correct}/{total})\n")
    print("Sample predictions:")
    for pred, true in examples:
        print(f"Predicted: {pred}")
        print(f"Actual: {true}\n")

    return avg_loss, exact_match_acc, examples


def inference(model, ink_file_path=None, ink_object=None, max_length=150, apply_correction=True):
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
    
    if apply_correction: latex_output = correct_latex(latex_output)
    
    return latex_output, ground_truth_latex, attention_weights



def main():
    print("Device:", DEVICE)

    data_root = download_data("https://storage.googleapis.com/mathwriting_data/mathwriting-2024.tgz")
    print("Data at:", data_root)
    full_train_ds = HMEDataset(data_root, "train")
    full_val_ds   = HMEDataset(data_root, "valid")
    full_test_ds  = HMEDataset(data_root, "test")
    print(f"Full sizes → train: {len(full_train_ds)}, valid: {len(full_val_ds)}, test: {len(full_test_ds)}")

    TRAIN_SUBSET_SIZE = 200000
    VAL_SUBSET_SIZE   = 15000
    TEST_SUBSET_SIZE  = 7000

    def sample(ds, size, seed):
        idxs = list(range(len(ds)))
        random.Random(seed).shuffle(idxs)
        return Subset(ds, idxs[:size])

    train_ds = sample(full_train_ds, TRAIN_SUBSET_SIZE, seed=42)
    val_ds   = sample(full_val_ds,   VAL_SUBSET_SIZE,   seed=43)
    test_ds  = sample(full_test_ds,  TEST_SUBSET_SIZE,  seed=44)

    print(f"Subset sizes → train: {len(train_ds)}, valid: {len(val_ds)}, test: {len(test_ds)}")
    # print("Running full train/valid/test set")
    train_loader = DataLoader(
        full_train_ds, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_variable_length_sequences
    )
    valid_loader = DataLoader(
        full_val_ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_variable_length_sequences
    )
    test_loader = DataLoader(
        full_test_ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_variable_length_sequences
    )

    feat0, _ = full_train_ds[0]
    input_dim = feat0.shape[1]
    encoder = Encoder(
        input_dim=input_dim, 
        proj_dim=128,           
        hidden_dim=256,         
        num_layers=2,           
        bidirectional=True,     
        dropout=0.3             
    )
    decoder = Decoder(
        output_dim=len(LATEX_VOCAB),
        embed_dim=128,          
        encoder_hidden_dim=256, 
        decoder_hidden_dim=256, 
        num_layers=2,           
        dropout=0.3,            
        num_heads=4             
    )
    model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)

    ckpt = "model_att/model_best_crc_full_data.pth"
    ckpt = "train model"
    if os.path.exists(ckpt):
        print("Loading checkpoint…")
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    else:
        print("Training new model…")
        train(model, train_loader, valid_loader)

    ink_path = os.path.join(data_root, "test/00c46c9b07b39bb7.inkml")
    print(f"\nExample inference on {ink_path}")
    
    print("Without correction:")
    pred_orig, gt, _ = inference(model, ink_file_path=ink_path, apply_correction=False)
    print(f"Predicted: {pred_orig}")
    print(f"Actual:    {gt}\n")
    
    print("With correction:")
    pred_fixed, _, _ = inference(model, ink_file_path=ink_path, apply_correction=True)
    print(f"Predicted: {pred_fixed}")
    print(f"Actual:    {gt}")
    print(f"Improvement: {'Yes' if pred_fixed == gt and pred_orig != gt else 'No'}")

    print("Full test:")
    test_model(
        model, test_loader,
        nn.CrossEntropyLoss(ignore_index=LATEX_PAD_TOKEN, label_smoothing=0.1)
    )

    
def indices_to_latex(indices, vocab_reverse):
    """Convert a sequence of token indices back to LaTeX string"""
    tokens = [vocab_reverse[idx] for idx in indices if idx in vocab_reverse and idx not in [LATEX_VOCAB['<pad>'], LATEX_VOCAB['<sos>'], LATEX_VOCAB['<eos>']]]
    return ''.join(tokens)


if __name__ == "__main__":
    main()
    
