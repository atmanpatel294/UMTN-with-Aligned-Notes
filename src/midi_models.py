import random
import pickle as pkl
import numpy as np

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

class MidiEncoder(nn.Module):
    def __init__(self, args, embeddings=None):
        super(MidiEncoder, self).__init__()
        self.input_size = args.input_embed_size
        self.hidden_size = args.encoder_hidden_size
        self.device = args.device
        if embeddings is None:
            self.embeddings = torch.randn(args.midi_vocab_size, args.input_embed_size)
        else:
            self.embeddings = embeddings
        # batch first!
        # bidirectional???
        self.lstm = nn.LSTM(args.input_embed_size, args.encoder_hidden_size)

    def forward(self, inputs):
        lstm_input = torch.matmul(inputs, self.embeddings) # (sl, bs, vocab_size) * (vocab_size, embed_size)
        _, (hidden, cell) = self.lstm(lstm_input)
        return hidden, cell

class MidiDecoder(nn.Module):
    def __init__(self, args, embeddings=None):
        super(MidiDecoder, self).__init__()
        self.input_size = args.input_embed_size
        self.hidden_size = args.decoder_hidden_size
        self.device = args.device
        if embeddings is None:
            self.embeddings = torch.randn(args.midi_vocab_size, args.input_embed_size)
        else:
            self.embeddings = embeddings
        self.lstm = nn.LSTM(args.input_embed_size, args.decoder_hidden_size)
        self.output = nn.Linear(args.decoder_hidden_size, args.midi_vocab_size)

    def init_cell(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size)

    def forward(self, encoder_output, ground_truth):
        seq_len, batch_size, vocab_size = ground_truth.size() # sl, bs, vs
        lstm_input = ground_truth[0]
        outputs = [lstm_input] # all the generated hidden layers and starting with SOS
        lstm_input = lstm_input.unsqueeze(0)
        cell = self.init_cell(batch_size) # 1, bs, hs
        hidden = encoder_output # 1, bs, hs

        for i in range(1, seq_len):
            expected = ground_truth[i].unsqueeze(0)
            _, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
            outputs.append(self.output(hidden).squeeze(0))
            lstm_input = expected

        return torch.stack(outputs)
