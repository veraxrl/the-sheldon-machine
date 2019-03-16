import torch
from torch import nn, autograd

from model_embeddings import ModelEmbeddings


class ConditionalLSTM(nn.Module):
    def __init__(self, vocab, embed_size, hidden_size, output_size, batch_size, dropout_rate=0.2):
        super(ConditionalLSTM, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        self.embedding = ModelEmbeddings(vocab, embed_size)
        self.context = nn.LSTM(embed_size, hidden_size, bidirectional=False)
        self.response = nn.LSTM(embed_size, hidden_size, bidirectional=False)
        self.proj = nn.Linear(hidden_size, output_size, bias=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.softmax = nn.LogSoftmax(dim=1)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        ## Need to modify for GNU training https://github.com/jiangqy/LSTM-Classification-Pytorch/blob/master/utils/LSTMClassifier.py
        h0 = autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_size))
        c0 = autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_size))
        return h0, c0

    def forward(self, context, response):
        # print(source.shape) # (batch, max_num_sents, max_num_words)
        context_embed = self.embedding(context)
        response_embed = self.embedding(response)
        context_embed = context_embed.transpose(0,1) #(max_num_sents, batch, embed_size)
        response_embed = response_embed.transpose(0,1) #(max_num_sents, batch, embed_size)

        output_context, self.hidden = self.context(context_embed, self.hidden)
        output_response, self.hidden = self.response(response_embed, self.hidden)
        ht, ct = self.hidden

        # print(ht.shape) = (1 * batch_size * hidden_size)
        # print(ht[-1].shape) = (batch_size * hidden_size)
        out = self.dropout(ht[-1])
        out2 = self.proj(out)
        out3 = self.softmax(out2)
        return out3
