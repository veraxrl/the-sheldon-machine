import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from model_embeddings import ModelEmbeddings

class LSTMClassifier(nn.Module):
    def __init__(self, vocab, embed_size, hidden_size, output_size, batch_size, dropout_rate=0.2):
        super(LSTMClassifier, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        self.embedding = ModelEmbeddings(vocab, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, bidirectional=False)
        self.proj = nn.Linear(hidden_size, output_size, bias=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.softmax = nn.LogSoftmax(dim=1)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        if torch.cuda.is_available():
            h0 = autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_size)).cuda()
            c0 = autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_size)).cuda()
        else: 
            h0 = autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_size))
            c0 = autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_size))
        return (h0, c0)

    def forward(self, source):
        # print(source.shape) # (batch, max_num_sents, max_num_words)
        x = self.embedding(source)
        x2 = x.transpose(0,1) #(max_num_sents, batch, embed_size)

        output, self.hidden = self.lstm(x2, self.hidden)
        ht, ct = self.hidden

        # print(ht.shape) = (1 * batch_size * hidden_size)
        # print(ht[-1].shape) = (batch_size * hidden_size)
        out = self.dropout(ht[-1])
        out2 = self.proj(out)
        out3 = self.softmax(out2)
        return out3

if __name__ == '__main__':
    c = LSTMClassifier(2, 2, 5, 10, 3)
    x = torch.Tensor([[1.0, 2.0],[3.0, 4.0]])
    c.forward(x, 5)