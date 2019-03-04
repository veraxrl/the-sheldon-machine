import torch
import torch.nn as nn
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

    def forward(self, source, target):
        print("***forward***")
        embed = self.embedding(source)
        packed_input = pack_padded_sequence(embed, lengths)
        output, (ht, ct) = self.lstm(packed_input, self.hidden_size)

        out = self.dropout(ht.sqeeze(0))
        out = self.proj(out)
        out = nn.Softmax(out)
        return out
        