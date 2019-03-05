import math
import sys
import pickle
import time

import numpy as np
from typing import List, Tuple, Dict, Set, Union
from readerUtils import read_discussion_forum, torch_from_json, generate_indices
from lstm import LSTMClassifier

from torch import nn, optim
import torch.utils.data as dataLoader
import torch.nn.utils
import torch.autograd as autograd

from sarcasmData import SarcasmData
from processing import DatasetProcessing

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
    context_dataset = DatasetProcessing(originals, responses, labels, "context")
    response_dataset = DatasetProcessing(originals, responses, labels, "response")
    #print(dataset.originals_idxs.shape)
    #print(dataset.responses_idxs.shape)
    #print(dataset.labels.view(len(labels), 1).shape)
    
    # expanding labels to be same dimensions so we can pack them together
    #context_data_set = (dataset.originals_idxs, dataset.labels.view(len(labels), 1).expand(len(labels), dataset.originals_idxs.size(1)))
    #response_data_set = (dataset.responses_idxs, dataset.labels.view(len(labels), 1).expand(len(labels), dataset.responses_idxs.size(1)))
    
    train_context_loader = dataLoader.DataLoader(context_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    train_response_loader = dataLoader.DataLoader(response_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    ### MAIN: 
    model = LSTMClassifier(word_vectors, embed_size, hidden_size, output_size, batch_size)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()
    train_loss = []
    test_loss = []
    train_acc = []
    test_acc = []
    for epoch in range(epochs):
        # Adjust learning rate using optimizer
        print("*"*10)
        print("Epoch #")
        print("*"*10)
        total_acc = 0.0
        total_loss = 0.0
        total = 0.0
        for iter, traindata in enumerate(train_response_loader):
            train_inputs, train_labels = traindata
            #print(train_inputs.shape)
            #print(train_labels.shape)

            model.zero_grad()
            model.hidden = model.init_hidden()
            output = model(train_inputs)
            #print(output.shape)
            loss = loss_function(output, train_labels)
            loss.backward()
            #optimizer.step()

            # Evaluation: acc calculation
            _, predicted = torch.max(output.data, 1)
            # print(predicted.tolist())
            # print(train_labels.tolist())
            # print("*")
            # print((predicted == train_labels))
            # print("*")
            total_acc += (predicted == train_labels).sum()
            # print(total_acc)
            total += len(train_labels)
            total_loss += loss.item()

            #TO-DO: write better logic to stop at last batch
            if iter > 600:
                break
        
        train_loss.append(total_loss/total)
        train_acc.append(total_acc.item()/total)
        print(np.mean(train_loss))
        print(np.mean(train_acc))


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
