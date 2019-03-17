import torch.nn as nn
import torch

class ModelEmbeddings(nn.Module): 
    '''
    Silimar to sentence-level embedding
    @Input: a batch of sentences (various length): [batch, max_sent_length]
    @Output: a batch of embeddings. One for each sentence: [batch, embed_size]
    '''
    def __init__(self, word_vectors, embed_size):
        super(ModelEmbeddings, self).__init__()
        """
        """
        self.embed_size = embed_size #300
        self.embedding = nn.Embedding.from_pretrained(word_vectors)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, input):
        """
        @input: Tensor of shape (batch, max_sent_length, max_word_length)
        @output: Tensor of shape (batch_size, embed_size)
        Averaged over max_sent_length
        """
        embedding = self.embedding(input)
        # print(embedding.shape)

        # Average over dim=2, keepdim=True
        lengths = (input > 1).sum(2, keepdim=True).float() + 1e-9
        emb = embedding.sum(2)
        avg_embed = emb / lengths

        # avg_embed = torch.mean(embedding, 2, True) #(batch_size, max_sent_length, embed_size)
        final_embed = self.dropout(torch.squeeze(avg_embed, 2))
        return final_embed
