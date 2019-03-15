import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from model_embeddings import ModelEmbeddings
from attentionLSTM import AttetionLSTM

class CombinedAttetionClassifier(nn.Module):
    def __init__(self, vocab, embed_size, hidden_size, output_size, batch_size, dropout_rate=0.2):
        super(CombinedAttetionClassifier, self).__init__()
        self.vocab = vocab
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size

        self.proj = nn.Linear(hidden_size*2, output_size, bias=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.softmax = nn.LogSoftmax(dim=1)

        self.LSTM_c = AttetionLSTM(self.vocab, self.embed_size, self.hidden_size, self.batch_size)
        self.LSTM_r = AttetionLSTM(self.vocab, self.embed_size, self.hidden_size, self.batch_size)

    def init_hidden(self):
        self.LSTM_c.hidden = self.LSTM_c.init_hidden()
        self.LSTM_r.hidden = self.LSTM_r.init_hidden()

    def forward(self, context, response):
        vc = self.LSTM_c(context)
        vr = self.LSTM_r(response)

        combined_v = torch.cat([vc,vr], dim=-1)
        output = self.proj(combined_v)
        output2 = self.softmax(output)
        output3 = self.dropout(output2)
        return output3
