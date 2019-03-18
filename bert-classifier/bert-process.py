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
                data.append([1 if row[1] == 'sarc' else 0, "a", row[4].strip()])
                line_count += 1
        print('Processed {} lines.'.format(line_count-1))
        return data

def write(data):
    threshold = int((len(data)-1) * 0.8)
    threshold2 = int((len(data)-1) * 0.9)
    # print(threshold)
    # print(threshold2)
    with open('./data/train.csv', mode='w') as train_file:
        for iter, row in enumerate(data[:threshold]):
            train_writer = csv.writer(train_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            train_writer.writerow([iter]+ row)

    with open('./data/dev.csv', mode='w') as train_file:
        for iter, row in enumerate(data[threshold:threshold2]):
            train_writer = csv.writer(train_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            train_writer.writerow([iter]+ row)

    with open('./data/test.csv', mode='w') as train_file:
        for iter, row in enumerate(data[threshold2:]):
            train_writer = csv.writer(train_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            if iter == 0:
                train_writer.writerow(["id", "sentence"])
            train_writer.writerow([iter]+ [row[2]])

def toTSV():
    df = pd.read_csv('./data/train.csv')
    df.to_csv('./data/train.tsv', sep='\t', index=False, header=False)
    df2 = pd.read_csv('./data/dev.csv')
    df2.to_csv('./data/dev.tsv', sep='\t', index=False, header=False)
    df3 = pd.read_csv('./data/test.csv')
    df3.to_csv('./data/test.tsv', sep='\t', index=False, header=True)


if __name__ == '__main__':
    data = read_discussion_forum()
    write(data)
    toTSV()