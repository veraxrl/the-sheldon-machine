import math
import sys
import pickle
import time

import numpy as np
from typing import List, Tuple, Dict, Set, Union
from tqdm import tqdm
from readerUtils import read_discussion_forum, torch_from_json
from lstm import SarcasmLSTM

import torch
import torch.nn.utils


def train(args: List):
    data = []
    # data source
    if 'discussion-forum' in args:
        data = read_discussion_forum()
        word_vectors = torch_from_json("./data/word_emb.json")
        print(word_vectors.shape)




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
