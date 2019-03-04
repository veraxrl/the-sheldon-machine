import math
import sys
import pickle
import time

import numpy as np
from typing import List, Tuple, Dict, Set, Union
from tqdm import tqdm
from readerUtils import readDiscussionForum

import torch
import torch.nn.utils


def train(args: List):
    data = []
    # data source
    if 'discussion-forum' in args:
        data = readDiscussionForum()




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
