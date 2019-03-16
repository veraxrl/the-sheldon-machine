import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from model_embeddings import ModelEmbeddings

class AttetionLSTM(nn.Module):
    def __init__(self, vocab, embed_size, hidden_size, batch_size):
        super(AttetionLSTM, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        self.embedding = ModelEmbeddings(vocab, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, bidirectional=False)
        self.s_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.tahn = nn.Tanh()
        self.softmax = nn.LogSoftmax(dim=1)

    def init_hidden(self):
        ## Need to modify for GNU training https://github.com/jiangqy/LSTM-Classification-Pytorch/blob/master/utils/LSTMClassifier.py
        h0 = autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_size))
        c0 = autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_size))
        return (h0, c0)

    def forward(self, source):
        #print(source.shape) # (batch, max_num_sents, max_num_words)
        x = self.embedding(source)
        x2 = x.transpose(0,1) #(max_num_sents, batch, embed_size)
        output, (ht, ct) = self.lstm(x2, self.hidden)
        output = output.view(self.batch_size, -1, self.hidden_size)
        #print(output.shape) #(batch, max_num_sents, hidden_size)
        #print(ht.shape) #(batch, 1, hidden_size)
        
        #Here: we want hidden states for all sents, not only the last hidden states to calculate attention
        u_i = self.tahn(self.s_proj(output)) #(max_num_sents, batch, hidden_size)
        u_s = torch.ones(self.batch_size, self.hidden_size, 1)
        e_t = torch.bmm(u_i, u_s).squeeze(2)
        #print(e_t.shape) #(batch, max_num_sents)
        alpha = self.softmax(e_t) #(batch, max_num_sents)
        alpha_view = (alpha.size(0), 1, alpha.size(1))
        v = torch.bmm(alpha.view(*alpha_view), output).squeeze(1) #(batch, hidden_state) 
        #v: document vector that summarizes all info in doc
        #print(v.shape)

        return v
