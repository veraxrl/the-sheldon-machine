import math
import sys
import pickle
import time

import numpy as np
from typing import List, Tuple, Dict, Set, Union
from readerUtils import read_discussion_forum, torch_from_json, generate_indices
from lstm import LSTMClassifier

from torch import nn, optim
import torch.nn.utils

from sarcasmData import SarcasmData

def train(args: List):
    ### LOAD EMBEDDINGS: 
    test_path = "./data/word_emb.json"
    word_vectors = torch_from_json(test_path)
    print(word_vectors.shape)

    ### PARAMETER SETTING:
    epochs = 10
    use_gpu = torch.cuda.is_available()
    learning_rate = 0.01
    vocab_size = word_vectors.size(0)
    embed_size = word_vectors.size(1)
    hidden_size = 256
    output_size = 2  #binary classification
    batch_size = 5

    ### DATA: 
    data = []
    if 'discussion-forum' in args:
        data = read_discussion_forum()
    originals, responses, labels = generate_indices(data)
    # dataset including all tensors
    dataset = SarcasmData(originals, responses, labels)
    print(dataset.originals_idxs.shape)
    print(dataset.responses_idxs.shape)
    print(dataset.labels.shape)

    ### MAIN: 
    model = LSTMClassifier(vocab_size, embed_size, hidden_size, output_size, batch_size)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()
    train_loss = []
    test_loss = []
    train_acc = []
    test_acc = []
    for epoch in range(epochs):
        # Adjust learning rate using optimizer
        print("one epoch")
        total_acc = 0.0
        total_loss = 0.0
        total = 0.0
        #for iter, traindata in enumerate(dataset.responses_idxs):


def train_model(model, optimizer, train, dev, x_to_ix, y_to_ix, batch_size, max_epochs):
    # The negative log likelihood loss. It is useful to train a classification problem with C classes.
    criterion = nn.NLLLoss()
    for epoch in range(max_epochs):
        print('Epoch:', epoch)
        y_true = list()
        y_pred = list()
        total_loss = 0
        for batch, targets, lengths, raw_data in utils.create_dataset(train, x_to_ix, y_to_ix, batch_size=batch_size):
            batch, targets, lengths = utils.sort_batch(batch, targets, lengths)
            model.zero_grad()
            pred, loss = apply(model, criterion, batch, targets, lengths)
            loss.backward()
            optimizer.step()
            
            pred_idx = torch.max(pred, 1)[1]
            y_true += list(targets.int())
            y_pred += list(pred_idx.data.int())
            total_loss += loss
        acc = accuracy_score(y_true, y_pred)
        val_loss, val_acc = evaluate_validation_set(model, dev, x_to_ix, y_to_ix, criterion)
        print("Train loss: {} - acc: {} \nValidation loss: {} - acc: {}".format(total_loss.data.float()/len(train), acc,
                                                                                val_loss, val_acc))
    return model

def decode(args: List):
    print("haha")


# sample command: python run.py train discussion-forum NMT
def main():
    """ Main func.
    """
    args = sys.argv
    
    if 'train' in args:
        train(args)
    elif 'decode' in args:
        decode(args)
    else:
        raise RuntimeError('invalid run mode')


if __name__ == '__main__':
    main()
