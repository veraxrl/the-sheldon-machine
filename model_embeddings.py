import torch.nn as nn

class ModelEmbeddings(nn.Module): 
    '''
    Silimar to sentence-level embedding
    @Input: a batch of sentences (various length): [batch, max_sent_length]
    @Output: a batch of embeddings. One for each sentence: [batch, embed_size]
    '''
    def __init__(self, vocab_size, embed_size):
        super(ModelEmbeddings, self).__init__()
        self.embed_size = embed_size #300
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, input):
        """
        @input: Tensor of shape (batch, max_sent_length)
        @output: Tensor of shape (batch_size, embed_size)
        Averaged over max_sent_length
        """
        embedding = self.embeddings(input) #(batch_size, max_sent_length, embed_size)
        #Average over dim=2, keepdim=True
        avg_embed = torch.mean(embedding, 2, True) #(batch_size, 1, embed_size)
        final_embed = self.dropout(torch.squeeze(avg_embed, 1))
        return final_embed
