from collections import Counter

import spacy as spacy
from tqdm import tqdm
import ujson as json
import sys
from readerUtils import readDiscussionForum
from typing import List


def save_word_vector(counter=None, emb_file="./data/glove.840B.300d.txt", vec_size=300, num_vectors=2196017):
    emb_mat, token2idx_dict = get_embedding(counter=counter, emb_file=emb_file, vec_size=vec_size, num_vectors=num_vectors)
    save('./data/word_emb.json', emb_mat, message="word embedding")
    save('./data/word2idx.json', token2idx_dict, message="word dictionary")


def get_embedding(counter=None, emb_file=None, vec_size=None, num_vectors=None):
    print("Pre-processing vectors...")
    embedding_dict = {}

    if emb_file is not None:
        assert vec_size is not None
        count = 0
        with open(emb_file, "r", encoding="utf-8") as fh:
            for line in tqdm(fh, total=num_vectors):
                array = line.split()
                word = "".join(array[0:-vec_size])
                vector = list(map(float, array[-vec_size:]))
                if counter is None or (counter is not None and word in counter and counter[word] > -1):
                    embedding_dict[word] = vector
        print("{} tokens have corresponding embedding vector".format(len(embedding_dict)))

    NULL = "--NULL--"
    OOV = "--OOV--"
    token2idx_dict = {token: idx for idx, token in enumerate(embedding_dict.keys(), 2)}
    token2idx_dict[NULL] = 0
    token2idx_dict[OOV] = 1
    embedding_dict[NULL] = [0. for _ in range(vec_size)]
    embedding_dict[OOV] = [0. for _ in range(vec_size)]
    idx2emb_dict = {idx: embedding_dict[token]
                    for token, idx in token2idx_dict.items()}
    emb_mat = [idx2emb_dict[idx] for idx in range(len(idx2emb_dict))]
    return emb_mat, token2idx_dict


def save(filename, obj, message=None):
    if message is not None:
        print("Saving {}...".format(message))
        with open(filename, "w") as fh:
            json.dump(obj, fh)


def populateCounter(counter: Counter, data: List):
    nlp = spacy.blank("en")
    for item in data:
        for sentence in [item[0], item[1]]:  # original and response
            for word in nlp(sentence):
                counter[word.text] += 1
    print("Counter populated with size {}".format(len(counter)))


# python setup.py discussion-forum
def main():
    """ Main func.
    """
    args = sys.argv

    data = []
    # if 'discussion-forum' in args:
    data = readDiscussionForum()
    counter = Counter()
    populateCounter(counter, data)
    save_word_vector(counter=counter)


if __name__ == '__main__':
    main()
