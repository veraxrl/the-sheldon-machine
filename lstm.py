import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size, batch_size, dropout_rate=0.2):
        super(LSTMClassifier, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.batch_size = batch_size

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, bidirectional=True)
        self.proj = nn.Linear(hidden_size, output_size, bias=True)
        self.dropout = nn.Dropout(dropout_rate)

    def init_hidden(self):
        ## Need to modify for GNU training https://github.com/jiangqy/LSTM-Classification-Pytorch/blob/master/utils/LSTMClassifier.py
        h0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim))
        c0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim))
        return (h0, c0)

    def forward(self, source, src_lengths):
        x = self.embedding(source)
        packed_input = pack_padded_sequence(x, src_lengths)
        output, (ht, ct) = self.lstm(packed_input, init_hidden())

        out = self.dropout(ht.sqeeze(0))
        out = self.proj(out)
        out = nn.Softmax(out)
        return out

if __name__ == '__main__':
    c = LSTMClassifier(300, 2, 5, 10, 3)
    x = torch.Tensor([[1.0, 2.0],[3.0, 4.0]])
    c.forward(x, 5)
