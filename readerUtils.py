import csv
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
            if line_count == 0:
                print('Column names are {}'.format(", ".join(row)))
                line_count += 1
            else:
                '''Remember to trim leading and trailing spaces'''
                data.append([row[3].strip(), row[4].strip(), 1 if row[2] == 'sarc' else 0])
                line_count += 1
        print('Processed {} lines.'.format(line_count))
        return data


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

    return tensor


def generate_indices(data: List, padding = True):
    '''Generate index for words. If the word doesn't exist, return 1 for that position'''

    nlp = spacy.blank("en")
    with open('./data/word2idx.json') as handle:
        word2idx = json.loads(handle.read())

    originals = []
    responses = []
    labels = []

    for original, response, label in data:
        originals.append([word2idx[word.text] if word.text in word2idx else 1 for word in nlp(original)])
        responses.append([word2idx[word.text] if word.text in word2idx else 1 for word in nlp(response)])
        labels.append(label)

    if padding:
        originals = add_padding(originals)
        responses = add_padding(responses)

    return originals, responses, labels


def add_padding(sents: List):
    '''Pad with 0'''
    max_len = np.max([len(s) for s in sents])
    for s in sents:
        s += (max_len - len(s)) * [0]
    return sents


def readReddit(file="reddit/train-balanced-sarcasm.csv"):
    # Import and data analysis
    print("-" * 80)
    print("Importing reddit training data")
    print("-" * 80)
    df = pd.read_csv(file)
    print(df.shape)
    print(df['label'].value_counts())
    # Spliting training and validation sets
    train_comment, valid_comment, train_label, valid_label = train_test_split(df['comment'], df['label'],
                                                                              train_size=0.8, test_size=0.2)


if __name__ == '__main__':
    readReddit()
