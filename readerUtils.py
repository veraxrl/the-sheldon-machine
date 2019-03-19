import csv
import pickle
import random

import numpy as np
import ujson as json
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import List
import spacy as spacy


'''
Read data from file and return a list of list
every item in the list is a list of original text, response text and label
label = 0 -> not sarcasm
label = 1 -> sarcasm
'''


def read_discussion_forum(file="./data/dicussion-forum-data.csv"):
    '''Read data from discussion forum data'''
    with open(file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        data = []
        for row in csv_reader:
            # if line_count > 100:
            #     break
            if line_count == 0:
                print('Column names are {}'.format(", ".join(row)))
                line_count += 1
            else:
                '''Remember to trim leading and trailing spaces'''
                data.append([row[3].strip(), row[4].strip(), 1 if row[1] == 'sarc' else 0])
                line_count += 1
        print('Processed {} lines.'.format(line_count))
        return data


'''
Stores indices to file in the format of [contexts, responses, labels]'''
def save_discussion_forum_data():
    data = read_discussion_forum()
    contexts, responses, labels = generate_indices(data, "discussion")
    with open('./data/discussion/discussion_forum_indices', 'wb') as f:
        pickle.dump([contexts, responses, labels], f)


def read_discussion_forum_from_file():
    with open('./data/discussion/discussion_forum_indices', 'rb') as f:
        my_list = pickle.load(f)
    d = list(zip(my_list[0], my_list[1], my_list[2]))
    random.shuffle(d)
    contexts, responses, labels = zip(*d)
    return contexts, responses, labels


def save_reddit_data():
    data = read_reddit_data()
    random.shuffle(data)
    contexts, responses, labels = generate_indices(data, "reddit")
    with open('./data/reddit/reddit_indices', 'wb') as f:
        pickle.dump([contexts, responses, labels], f)


def read_reddit_data_from_file():
    with open('./data/reddit/reddit_indices', 'rb') as f:
        my_list = pickle.load(f)
    d = list(zip(my_list[0], my_list[1], my_list[2]))
    random.shuffle(d)
    contexts, responses, labels = zip(*d)
    return contexts, responses, labels

def torch_from_json(path, dtype=torch.float32):
    """Load a PyTorch Tensor from a JSON file.

    Args:
        path (str): Path to the JSON file to load.
        dtype (torch.dtype): Data type of loaded array.

    Returns:
        tensor (torch.Tensor): Tensor loaded from JSON file.
    """

    with open(path, 'r') as fh:
        array = np.array(json.load(fh))

    tensor = torch.from_numpy(array).type(dtype)
    if torch.cuda.is_available():
        tensor = tensor.cuda()

    return tensor


'''
Separate every content to sentences, and generate index for each word. Pad the sentence and contents.
sents: a list of contents
Every content is consisted of a list: [original sentences, response sentences, label]
'''
def generate_indices(sents: List, tag):
    '''Generate index for words. If the word doesn't exist, return 1 for that position'''

    nlp = spacy.load("en_core_web_sm")
    if tag == "discussion":
        word2idxPath = './data/discussion/word2idx.json'
    else:
        word2idxPath = './data/reddit/word2idx.json'

    with open('./data/discussion/word2idx.json') as handle:
            word2idx = json.loads(handle.read())

    originals = []
    responses = []
    labels = []

    count = 0

    for content in sents:
        # separate every content into a list of sentences
        original = [sentence.text for sentence in nlp(content[0]).sents]
        response = [sentence.text for sentence in nlp(content[1]).sents]
        label = content[2]
        originals.append([[word2idx[word.text] if word.text in word2idx else 1 for word in nlp(sentence)] for sentence in original])
        responses.append([[word2idx[word.text] if word.text in word2idx else 1 for word in nlp(sentence)] for sentence in response])
        labels.append(label)
        count += 1
        if count % 1000 is 0:
            print("At count {}".format(count))
            originals = pad_sents(originals)
            responses = pad_sents(responses)
            with open('./data/reddit/reddit_indices', 'wb') as f:
                pickle.dump([originals, responses, labels], f)
            # if count > 8000:
            #     break
    originals = pad_sents(originals)
    responses = pad_sents(responses)
    # print(originals)
    # print(responses)
    print("Generated indices for contexts and responses.")
    return originals, responses, labels


'''
@return: A list of content padded
Every content has max_content_length of sentences
Every sentence has max_sentence_length of words
'''
def pad_sents(sents: List):
    # max_content_length = np.max([len(content) for content in sents])
    # max_sentence_length = [len(sentence) for sentence in content for content in sents]
    max_content_length = 10
    max_sentence_length = 50
    for content in sents:
        if len(content) > max_content_length:
            del content[max_content_length:]
        else:
            content += (max_content_length - len(content)) * [max_sentence_length * [0]]

        for sentence in content:
            if len(sentence) > max_sentence_length:
                del sentence[max_sentence_length:]
            else:
                sentence += (max_sentence_length - len(sentence)) * [0]
    return sents


def read_reddit_data(file="./data/reddit/train-balanced-sarcasm.csv"):
    with open(file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        data = []
        for row in csv_reader:
            # if line_count > 1000:
            #     break
            if line_count == 0:
                print('Column names are {}'.format(", ".join(row)))
                line_count += 1
            else:
                '''Remember to trim leading and trailing spaces'''
                data.append([row[9].strip(), row[1].strip(), 1 if row[0] == '1' else 0])
                line_count += 1
        print('Processed {} lines.'.format(line_count))
        return data


if __name__ == '__main__':
    # read_discussion_forum()
    # save_discussion_forum_data()
    save_reddit_data()
    # with open('./data/discussion/discussion_forum_indices', 'rb') as f:
    #     my_list = pickle.load(f)
    #     print(my_list)
