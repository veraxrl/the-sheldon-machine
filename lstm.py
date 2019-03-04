import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size, dropout_rate=0.2):
        super(SarcasmLSTM, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, bidirectional=True)
        self.proj = nn.Linear(hidden_size, output_size, bias=True)
        self.dropout = nn.Dropout(self.dropout_rate)

    def init_hidden(self, batch_size):
        return(autograd.Variable(torch.randn(1, batch_size, self.hidden_dim)), autograd.Variable(torch.randn(1, batch_size, self.hidden_dim)))

    def forward(self, source, src_lengths):
        print("***forward***")
        x = self.embedding(source)
        packed_input = pack_padded_sequence(x, src_lengths)
        output, (ht, ct) = self.lstm(packed_input, init_hidden(source.size(-1)))

        out = self.dropout(ht.sqeeze(0))
        out = self.proj(out)
        out = nn.Softmax(out)
        return out
        