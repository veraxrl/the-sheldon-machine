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

    def init_hidden(self):
        ## Need to modify for GNU training https://github.com/jiangqy/LSTM-Classification-Pytorch/blob/master/utils/LSTMClassifier.py
        h0 = autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_size))
        c0 = autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_size))
        return (h0, c0)

    def forward(self, source):
        #print(source.shape)
        x = self.embedding(source).view(1, self.batch_size, self.embed_size)
        #print(x.shape) #  (batch, input_size) -> (1, batch, input_size)
        output, (ht, ct) = self.lstm(x, self.init_hidden())

        out = self.dropout(torch.squeeze(ht, 0))
        out = self.proj(out) #[2, batch_size, output_size] -> bidirectional? Is this right?
        out = F.log_softmax(out)
        return out

if __name__ == '__main__':
    c = LSTMClassifier(2, 2, 5, 10, 3)
    x = torch.Tensor([[1.0, 2.0],[3.0, 4.0]])
    c.forward(x, 5)
