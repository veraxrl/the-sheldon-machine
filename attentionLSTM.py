import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from model_embeddings import ModelEmbeddings

class AttentionLSTM(nn.Module):
    def __init__(self, vocab, embed_size, hidden_size, batch_size):
        super(AttentionLSTM, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        self.embedding = ModelEmbeddings(vocab, embed_size)
        #self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(embed_size, hidden_size, bidirectional=False)
        self.s_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.tahn = nn.Tanh()
        self.u_s = nn.Parameter(torch.ones(self.batch_size, self.hidden_size, 1))

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
        #print(ht.shape) #(1, batch, hidden_size)
        
        #Here: we want hidden states for all sents, not only the last hidden states to calculate attention
        u_i = self.tahn(self.s_proj(output)) #(batch, max_num_sents hidden_size)
        #Alternatively using last hidden states in attention cal: u_s = ht.view(self.batch_size, self.hidden_size, 1)
        e_t = torch.bmm(u_i, self.u_s).squeeze(2)
        #print(e_t.shape) #(batch, max_num_sents)
        alpha = F.softmax(e_t, dim=1) #(batch, max_num_sents)
        v = torch.bmm(output.transpose(1,2), alpha.unsqueeze(2)).squeeze(2)
        #v: document vector that summarizes all info in doc
        #print(v.shape)

        return v
