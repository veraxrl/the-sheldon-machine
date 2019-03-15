import math
import sys
import pickle
import time

import numpy as np
from typing import List, Tuple, Dict, Set, Union
from readerUtils import read_discussion_forum, torch_from_json, generate_indices, read_reddit_data, \
    read_discussion_forum_from_file
from combinedLSTM import CombinedAttetionClassifier

from torch import nn, optim
import torch.utils.data as dataLoader
import torch.nn.utils
import torch.autograd as autograd

from sarcasmData import SarcasmData
from concatDataset import ConcatDataset

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
        originals, responses, labels = read_discussion_forum_from_file()
    elif 'reddit' in args:
        data = read_reddit_data()

    # dataset including all tensors
    concat_dataset = ConcatDataset(originals, responses, labels)
    train_concat_loader = dataLoader.DataLoader(concat_dataset, batch_size=batch_size, num_workers=4)

    ### MAIN:
    model = CombinedAttetionClassifier(word_vectors, embed_size, hidden_size, output_size, batch_size)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    loss_function = nn.NLLLoss(size_average=False)
    train_loss = []
    test_loss = []
    train_acc = []
    test_acc = []
    right_sar = 0
    all_sar = 0
    for epoch in range(epochs):
        # Adjust learning rate using optimizer
        print("*"*10)
        print("Epoch #{}".format(epoch))
        print("*"*10)
        total_acc = 0.0
        total_loss = 0.0
        total = 0.0
        for iter, traindata in enumerate(train_concat_loader):
            train_data, train_labels = traindata
            train_context_inputs = train_data[0]
            train_response_inputs= train_data[1]
            # print(train_context_inputs.shape)
            # print(train_response_inputs.shape)
            # print(train_label.shape)

            optimizer.zero_grad()
            model.hidden = model.init_hidden()
            output = model(train_context_inputs, train_response_inputs)
            #print(output.shape)
            loss = loss_function(output, train_labels)
            loss.backward()
            optimizer.step()

            # Evaluation: accuracy and precision calculation
            _, predicted = torch.max(output.data, 1)
            gold_labels = train_labels.tolist()
            correct_labels = (predicted == train_labels).tolist()
            # print(gold_labels)
            # print(predicted.tolist())
            # print(correct_labels)
            # print()
            all_sar += np.sum(gold_labels)
            for i, clabel in enumerate(correct_labels):
                if clabel == 1 and gold_labels[i] == 1: #correct sarcasm detection
                    right_sar += 1

            total_acc += (predicted == train_labels).sum()
            # print(total_acc)
            total += len(train_labels)
            total_loss += loss.item()

            #TO-DO: write better logic to stop at last batch
            if iter > 800:
                break
        
        # print(right_sar) #precision
        # print(all_sar)
        train_loss.append(total_loss/total)
        train_acc.append(total_acc.item()/total)
        print("Sarcasm precision is {}".format(right_sar/all_sar))  #sarcasm precision
        print(np.mean(train_loss))
        print(np.mean(train_acc))

# sample command: python run.py train discussion-forum NMT
def main():
    """ Main func.
    """
    args = sys.argv
    
    if 'train' in args:
        train(args)
    else:
        raise RuntimeError('invalid run mode')


if __name__ == '__main__':
    main()
