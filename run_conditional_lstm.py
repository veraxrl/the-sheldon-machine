import sys
import numpy as np
from typing import List, Tuple, Dict, Set, Union

import sklearn
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from conditionalLSTM import ConditionalLSTM
from readerUtils import read_discussion_forum, torch_from_json, generate_indices, read_reddit_data, \
    read_discussion_forum_from_file, read_reddit_data_from_file

from torch import nn, optim
import torch.utils.data as dataLoader
import torch.nn.utils
from processing import DatasetProcessing


### PARAMETER SETTING:
epochs = 30
use_gpu = torch.cuda.is_available()
print("Use GPU is {}".format(use_gpu))
learning_rate = 0.01
hidden_size = 128
output_size = 2  # binary classification
batch_size = 10


def train(args: List):
    ### LOAD EMBEDDINGS:
    if 'discussion-forum' in args:
        test_path = "./data/discussion/word_emb.json"
    elif 'reddit' in args:
        test_path = "./data/reddit/word_emb.json"

    word_vectors = torch_from_json(test_path)
    print(word_vectors.shape)

    vocab_size = word_vectors.size(0)
    embed_size = word_vectors.size(1)

    data_map = prepare_data(args)

    model = train_model(word_vectors, embed_size, data_map)

    evaluate_test(model, data_map)


def prepare_data(args: List):
    data_map = {}
    if 'discussion-forum' in args:
        originals, responses, labels = read_discussion_forum_from_file()

    elif 'reddit' in args:
        originals, responses, labels = read_reddit_data_from_file()

    # split train and test
    threshold = int(len(originals) * 0.8)
    data_map['train_context'] = build_data_loader(originals[:threshold], responses[:threshold], labels[:threshold],
                                                  'context')
    data_map['train_response'] = build_data_loader(originals[:threshold], responses[:threshold], labels[:threshold],
                                                   'response')
    data_map['train_label'] = labels[:threshold]
    data_map['test_context'] = build_data_loader(originals[threshold:], responses[threshold:], labels[threshold:],
                                                 'context')
    data_map['test_response'] = build_data_loader(originals[threshold:], responses[threshold:], labels[threshold:],
                                                  'response')
    data_map['test_label'] = labels[threshold:]

    return data_map


def train_model(word_vectors, embed_size, data_map):
    ### MAIN:
    model = ConditionalLSTM(word_vectors, embed_size, hidden_size, output_size, batch_size)

    if use_gpu:
        model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # loss_function = nn.CrossEntropyLoss()
    loss_function = nn.NLLLoss()
    train_loss = []
    test_loss = []
    train_acc = []
    test_acc = []
    right_sar = 0
    all_sar = 0

    train_response_loader = data_map.get('train_response')
    train_context_loader = data_map.get('train_context')

    for epoch in range(epochs):
        model.train()

        # Adjust learning rate using optimizer
        print("*" * 10)
        print("Epoch #{}".format(epoch))
        print("*" * 10)
        total_acc = 0.0
        total_loss = 0.0
        total = 0.0

        for x in zip(enumerate(train_context_loader), enumerate(train_response_loader)):
            context_inputs = x[0][1][0]
            response_inputs = x[1][1][0]
            train_labels = x[0][1][1]

            if use_gpu:
                context_inputs = context_inputs.cuda()
                response_inputs = response_inputs.cuda()
                train_labels = train_labels.cuda()

            if context_inputs.size()[0] < batch_size:
                break

            optimizer.zero_grad()
            model.hidden = model.init_hidden()
            output = model(context_inputs, response_inputs)
            # print(output.shape)
            loss = loss_function(output, train_labels)
            loss.backward()
            optimizer.step()

            total += len(train_labels)
            total_loss += loss.item()

        train_loss.append(total_loss / total)
        print("Loss is {}".format(np.mean(train_loss)))
        evaluate_test(model, data_map)
        print()

    return model


def build_data_loader(originals: List, responses: List, labels: List, type):
    dataset = DatasetProcessing(originals, responses, labels, type)
    loader = dataLoader.DataLoader(dataset, batch_size=batch_size, num_workers=4)
    return loader
'''
# dataset including all tensors
    context_dataset = DatasetProcessing(originals, responses, labels, "context")
    response_dataset = DatasetProcessing(originals, responses, labels, "response")
    # print(dataset.originals_idxs.shape)
    # print(dataset.responses_idxs.shape)

    # expanding labels to be same dimensions so we can pack them together
    # context_data_set = (dataset.originals_idxs, dataset.labels.view(len(labels), 1).expand(len(labels), dataset.originals_idxs.size(1)))
    # response_data_set = (dataset.responses_idxs, dataset.labels.view(len(labels), 1).expand(len(labels), dataset.responses_idxs.size(1)))

    # ctx_sampler = dataLoader.RandomSampler(context_dataset, False)
    # res_sampler = dataLoader.RandomSampler(response_dataset, False)
    train_context_loader = dataLoader.DataLoader(context_dataset, batch_size=batch_size, num_workers=4)
    train_response_loader = dataLoader.DataLoader(response_dataset, batch_size=batch_size, num_workers=4)
    return train_context_loader, train_response_loader, labels
'''


def evaluate_test(model, data_map):

    model.eval()

    test_response_loader = data_map['test_response']
    test_context_loader = data_map['test_context']

    all_pred_labels = []

    for x in zip(enumerate(test_context_loader), enumerate(test_response_loader)):
        context_inputs = x[0][1][0]
        response_inputs = x[1][1][0]

        if use_gpu:
            context_inputs = context_inputs.cuda()
            response_inputs = response_inputs.cuda()

        if context_inputs.size()[0] < batch_size:
            break

        pred_data = model(context_inputs, response_inputs)
        pred_labels = torch.max(pred_data, 1)[1].cpu()
        all_pred_labels.extend(pred_labels)

    # make sure gold_labels is truncated to the same length as all_pred_labels
    gold_labels = data_map['test_label'][:len(all_pred_labels)]
    evaluation = precision_recall_fscore_support(gold_labels, all_pred_labels)
    precision = evaluation[0]
    recall = evaluation[1]
    fscore = evaluation[2]
    accuracy = accuracy_score(gold_labels, all_pred_labels)
    print("Precision score for evaluation is {}".format(precision))
    print("Recall score for evaluation is {}".format(recall))
    print("F1 score for evaluation is {}".format(fscore))
    print("Accuracy score for evaluation is {}".format(accuracy))


# sample command: python run.py train discussion-forum NMT
def main():
    """ Main func.
    """
    args = sys.argv

    train(args)


if __name__ == '__main__':
    main()
