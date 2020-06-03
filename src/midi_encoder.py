import random
import pickle as pkl
import numpy as np

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MyLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MyLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc_f = nn.Linear(hidden_size+input_size, hidden_size)
        self.fc_i = nn.Linear(hidden_size+input_size, hidden_size)
        self.fc_o = nn.Linear(hidden_size+input_size, hidden_size)
        self.fc_c = nn.Linear(hidden_size+input_size, hidden_size)
        
    def forward(self, x, h_prev, c_prev):
        h_prev, x = h_prev.float(), x.float()    
        inputs = torch.cat((h_prev,x),1)
        f = torch.sigmoid(self.fc_f(inputs))
        i = torch.sigmoid(self.fc_i(inputs))
        o = torch.sigmoid(self.fc_o(inputs))
        c_dash = torch.tanh(self.fc_c(inputs))
        c_new = f*c_prev + i*c_dash
        h_new = o * torch.tanh(c_new)
        return h_new, c_new

class MidiEncoder(nn.Module):
    def __init__(self, args):
        super(MidiEncoder, self).__init__()
        self.vocab_size = args.m_vocab_size
        self.hidden_size = args.m_hidden_size
        self.embeddings = nn.Embedding(args.m_vocab_size, args.m_embed_size)
        self.lstm = MyLSTMCell(args.m_embed_size+1, args.m_hidden_size)

    def forward(self, midi_chords, midi_durations):
        batch_size, seq_len = midi_chords.size()
        hidden = self.init_hidden(batch_size)
        cell = self.init_hidden(batch_size)
        embed = self.embeddings(midi_chords)
                
        for i in range(seq_len):
            # print("embed: ", embed[:,i,:].size())
            # print("durations: ", torch.unsqueeze(midi_durations[:,i], dim=1).size())
            lstm_input = torch.cat((embed[:,i,:], torch.unsqueeze(midi_durations[:,i], dim=1)),dim=1) #(bs, em_size) (bs)
            hidden, cell = self.lstm.forward(lstm_input, hidden, cell)
        return hidden, cell

    def init_hidden(self, bs):
        return torch.zeros(bs, self.hidden_size).type(torch.FloatTensor).cuda()

# class Decoder(nn.Module):
#     def __init__(self, vocab_size, hidden_size):
#         super(Decoder, self).__init__()
#         self.vocab_size = vocab_size
#         self.hidden_size = hidden_size
#         self.embeddings = nn.Embedding(vocab_size, opts.embed_size) #CHANGE 1
#         self.lstm = MyLSTMCell(opts.embed_size, hidden_size) #CHANGE 1
#         self.output = nn.Linear(opts.hidden_layer_size, opts.vocab_size) #torch.randn(batch_size, vocab)

#     def forward(self, x, h_prev, c_prev):
#         lstm_input = self.embeddings(x).squeeze(1)
#         h_new, c_new = self.lstm.forward(lstm_input, h_prev, c_prev)
#         output = self.output(h_new)
#         return output, h_new, c_new

