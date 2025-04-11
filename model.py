import torch
import torch.nn as nn
import torch.nn.functional as F


import random
import subprocess


from config import *

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, bidirectional=True):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, 
                            bidirectional=bidirectional, batch_first=True)
        
    def forward(self, x, lengths):
        # pack the padded sequences for the LSTM
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        o, (h, c) = self.lstm(packed)
        o, _ = nn.utils.rnn.pad_packed_sequence(o, batch_first=True)
        return o, (h, c)
    
class Attention(nn.Module):
    def __init__(self, encoder_hidden_dim, decoder_hidden_dim):
        super(Attention, self).__init__()
        # encoder is bidirectional, so its hidden dimension is doubled
        self.attn = nn.Linear((encoder_hidden_dim * 2) + decoder_hidden_dim, decoder_hidden_dim)
        self.v = nn.Parameter(torch.rand(decoder_hidden_dim))

    def forward(self, hidden, encoder_outputs, mask):
        # hidden: (batch, decoder_hidden_dim)
        # encoder_outputs: (batch, seq_len, encoder_hidden_dim*2)
        batch_size = encoder_outputs.shape[0]
        seq_len = encoder_outputs.shape[1]
    
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))  # (batch, seq_len, decoder_hidden_dim)
        energy = energy.transpose(1, 2)  # (batch, decoder_hidden_dim, seq_len)
        v = self.v.repeat(batch_size, 1).unsqueeze(1)  # (batch, 1, decoder_hidden_dim)
        attn_weights = torch.bmm(v, energy).squeeze(1)  # (batch, seq_len)
        attn_weights = attn_weights.masked_fill(mask == 0, -1e10)
        return F.softmax(attn_weights, dim=1)
    
class Decoder(nn.Module):
    def __init__(self, output_dim, embed_dim, encoder_hidden_dim, decoder_hidden_dim, num_layers=1):
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, embed_dim)
        # input to the LSTM will be the embedding concatenated with the context vector from attention
        self.lstm = nn.LSTM(embed_dim + encoder_hidden_dim * 2, decoder_hidden_dim, num_layers=num_layers, batch_first=True)
        self.attention = Attention(encoder_hidden_dim, decoder_hidden_dim)
        # output layer takes the LSTM output, context vector, and embedding to predict the next token
        self.fc_out = nn.Linear(decoder_hidden_dim + encoder_hidden_dim * 2 + embed_dim, output_dim)
    
    def forward(self, input, hidden, cell, encoder_outputs, mask):
        # input: (batch,) current token indices
        input = input.unsqueeze(1)  # (batch, 1)
        embedded = self.embedding(input)  # (batch, 1, embed_dim)
        
        # attention weights and context vector from encoder outputs
        attn_weights = self.attention(hidden[-1], encoder_outputs, mask)  # (batch, seq_len)
        attn_weights = attn_weights.unsqueeze(1)  # (batch, 1, seq_len)
        context = torch.bmm(attn_weights, encoder_outputs)  # (batch, 1, encoder_hidden_dim*2)
        
        # embedded input and context vector
        lstm_input = torch.cat((embedded, context), dim=2)  # (batch, 1, embed_dim + encoder_hidden_dim*2)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        
        output = output.squeeze(1)
        context = context.squeeze(1)
        embedded = embedded.squeeze(1)
        # next token
        prediction = self.fc_out(torch.cat((output, context, embedded), dim=1))  # (batch, output_dim)
        return prediction, hidden, cell, attn_weights

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
