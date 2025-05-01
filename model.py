import torch
import torch.nn as nn
import torch.nn.functional as F

import random
import subprocess

from config import *

class Encoder(nn.Module):
    def __init__(self, input_dim, proj_dim=64, hidden_dim=128, num_layers=2, bidirectional=True, dropout=0.3):
        super(Encoder, self).__init__()
        self.num_layers    = num_layers
        self.bidirectional = bidirectional
        self.proj = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(
            proj_dim, hidden_dim, num_layers=num_layers,
            bidirectional=bidirectional, batch_first=True,
            dropout=dropout if num_layers>1 else 0
        )

    def forward(self, x, lengths):
        # x: (B, T, input_dim)
        x = self.proj(x)  # (B, T, proj_dim)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        o, (h, c) = self.lstm(packed)
        o, _ = nn.utils.rnn.pad_packed_sequence(o, batch_first=True)
        return o, (h, c)
    
# class Attention(nn.Module):
#     def __init__(self, encoder_hidden_dim, decoder_hidden_dim):
#         super(Attention, self).__init__()
#         # encoder is bidirectional, so its hidden dimension is doubled
#         self.attn = nn.Linear((encoder_hidden_dim * 2) + decoder_hidden_dim, decoder_hidden_dim)
#         self.v = nn.Parameter(torch.rand(decoder_hidden_dim))

#     def forward(self, hidden, encoder_outputs, mask):
#         # hidden: (batch, decoder_hidden_dim)
#         # encoder_outputs: (batch, seq_len, encoder_hidden_dim*2)
#         batch_size = encoder_outputs.shape[0]
#         seq_len = encoder_outputs.shape[1]
    
#         hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)
#         energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))  # (batch, seq_len, decoder_hidden_dim)
#         energy = energy.transpose(1, 2)  # (batch, decoder_hidden_dim, seq_len)
#         v = self.v.repeat(batch_size, 1).unsqueeze(1)  # (batch, 1, decoder_hidden_dim)
#         attn_weights = torch.bmm(v, energy).squeeze(1)  # (batch, seq_len)
#         attn_weights = attn_weights.masked_fill(mask == 0, -1e10)
#         return F.softmax(attn_weights, dim=1)

class MultiHeadAttention(nn.Module):
    def __init__(self, enc_hidden_dim, dec_hidden_dim, num_heads=4, dropout=0.3):
        super().__init__()
        self.enc_dim = enc_hidden_dim * 2
        self.dec_dim = dec_hidden_dim

        # project decoder hidden state into encoder‐dim for queries
        self.proj_q = nn.Linear(dec_hidden_dim, self.enc_dim)
        # multi‐head attention in encoder hidden space
        self.mha = nn.MultiheadAttention(
            embed_dim=self.enc_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden, encoder_outputs, mask):
        # hidden:           (batch, dec_hidden_dim)
        # encoder_outputs:  (batch, seq_len, enc_dim)
        # mask:             (batch, seq_len)  boolean (True=valid)

        # 1) project query
        q = self.proj_q(hidden)           # (batch, enc_dim)
        q = q.unsqueeze(1)                # (batch, 1, enc_dim)

        # 2) multi-head attention
        # key_padding_mask expects True==pad, so invert mask
        key_pad = ~mask                   # (batch, seq_len)
        attn_output, attn_weights = self.mha(
            query=q,
            key=encoder_outputs,
            value=encoder_outputs,
            key_padding_mask=key_pad
        )
        # attn_weights: (batch, 1, seq_len)
        attn_weights = attn_weights.squeeze(1)  # (batch, seq_len)
        context = attn_output.squeeze(1)        # (batch, enc_dim)
        return attn_weights, self.dropout(context)
    
class Decoder(nn.Module):
    def __init__(self, output_dim, embed_dim, encoder_hidden_dim, decoder_hidden_dim, num_layers=2, dropout=0.3, num_heads=4):
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, embed_dim)
        self.embed_drop = nn.Dropout(dropout)
        
        self.lstm = nn.LSTM(embed_dim + encoder_hidden_dim * 2, decoder_hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers>1 else 0)
        self.attention = MultiHeadAttention(encoder_hidden_dim, decoder_hidden_dim, num_heads=num_heads, dropout=dropout)
        
        self.fc_out = nn.Sequential(
            nn.Linear(decoder_hidden_dim + encoder_hidden_dim*2 + embed_dim, decoder_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(decoder_hidden_dim, output_dim)
        )

        self.layer_norm = nn.LayerNorm(output_dim)
    
    def forward(self, input, hidden, cell, encoder_outputs, mask):
        # input: (batch,) current token
        emb = self.embedding(input)      # (batch, embed_dim)
        emb = self.embed_drop(emb)
        emb = emb.unsqueeze(1)           # (batch, 1, embed_dim)

        # attention
        top_hidden = hidden[-1]          # (batch, dec_hidden_dim)
        attn_w, context = self.attention(top_hidden, encoder_outputs, mask)
        context = context.unsqueeze(1)   # (batch, 1, enc_dim)

        # LSTM step
        lstm_in = torch.cat((emb, context), dim=2)  # (batch,1,embed+enc_dim)
        out, (hidden, cell) = self.lstm(lstm_in, (hidden, cell))
        out = out.squeeze(1)              # (batch, dec_hidden_dim)

        # predict next token
        context = context.squeeze(1)      # (batch, enc_dim)
        emb     = emb.squeeze(1)          # (batch, embed_dim)
        combined = torch.cat((out, context, emb), dim=1)  # (batch, dec+enc+emb)
        pred = self.fc_out(combined)      # (batch, output_dim)
        return self.layer_norm(pred), hidden, cell, attn_w

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def create_mask(self, src):
        # mask to ignore padding (assuming padded values are all zeros)
        # src shape: (batch, seq_len, feature_dim)
        mask = (src.sum(dim=2) != 0)  # (batch, seq_len)
        return mask

    def forward(self, src, src_lengths, trg, teacher_forcing_ratio=0.5):
        # src: (batch, src_seq_len, feature_dim)
        # trg: (batch, trg_seq_len) where each element is a token index
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim
        
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        
        encoder_outputs, (hidden, cell) = self.encoder(src, src_lengths)
        
        # if the encoder is bidirectional, combine the two directions for each layer
        if self.encoder.bidirectional:
            # hidden: (num_layers*2, batch, hidden_dim) -> reshape to (num_layers, 2, batch, hidden_dim)
            hidden = hidden.view(self.encoder.num_layers, 2, hidden.size(1), hidden.size(2)).sum(dim=1)
            cell = cell.view(self.encoder.num_layers, 2, cell.size(1), cell.size(2)).sum(dim=1)
        
        # mask for attention
        mask = self.create_mask(src)
        
        # first input to the decoder is the <sos> token (index 0)
        input_token = trg[:, 0]
        
        for t in range(1, trg_len):
            o, h, c, _ = self.decoder(input_token, hidden, cell, encoder_outputs, mask)
            outputs[:, t] = o
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = o.argmax(1)
            input_token = trg[:, t] if teacher_force else top1
        
        return outputs
