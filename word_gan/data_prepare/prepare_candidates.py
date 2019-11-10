import json
import os

import gensim

from word_gan.settings import SETTINGS

TARGET_PATH = os.path.join(SETTINGS.VOCAB_PATH, 'target.txt')
MODEL_PATH = os.path.join(SETTINGS.DATA_DIR, 'target_vectors.txt')
CANDIDATES_OUT_PATH = os.path.join(SETTINGS.DATA_DIR, 'candidates.json')
TOPN = SETTINGS.NUM_VARIANTS


def load_vocab(path):
    vocab = []
    with open(path) as fd:
        for line in fd:
            line = line.strip()
            if line != '@@UNKNOWN@@':
                vocab.append(line)

    return vocab


def build_candidates(vocab):
    """

    >>> build_candidates(['run'])

    :param vocab:
    :return:
    """
    model = gensim.models.KeyedVectors.load_word2vec_format(MODEL_PATH)

    mapping = {}

    for word in vocab:
        candidates = [x[0] for x in model.most_similar(positive=[word], topn=TOPN)]
        mapping[word] = candidates

    return mapping


if __name__ == '__main__':
    vocab = load_vocab(TARGET_PATH)
    candidates = build_candidates(vocab)

    with open(CANDIDATES_OUT_PATH, 'w') as out:
        json.dump(candidates, out, indent=2)
