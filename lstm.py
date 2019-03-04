import torch
import torch.nn as nn
import torch.nn.functional as F

class SarcasmLSTM(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, output_size, dropout_rate=0.2):
        super(SarcasmLSTM, self).__init__()
        self.embedding = nn.Embedding(input_size, embed_size)
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(embed_size, hidden_size, bidirectional=True)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.proj = nn.Linear(hidden_size, output_size, bias=True)

    def forward(self, source, target):
        print("***forward***")
        embedded = self.embedding(source)
        output, hidden = self.lstm(embedded)
        out2 = self.proj(hidden.squeeze(0))
        P = F.log_softmax(out2)
        return P
        