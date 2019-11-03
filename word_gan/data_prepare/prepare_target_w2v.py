import os
from typing import List

import pandas as pd
from gensim.models import KeyedVectors

from word_gan.settings import SETTINGS


def create_w2v_from_fasttext(
        fasttext_model_path: str,
        vocab_path: str,
        w2v_out_path: str
):
    model = KeyedVectors.load(fasttext_model_path)

    vocab = set()
    with open(vocab_path) as fd:
        for word in fd:
            vocab.add(word.strip().lower())

    with open(w2v_out_path, 'w') as out:
        out.write(f"{len(vocab)} {model.vector_size}\n")
        for word in vocab:
            word_vector_txt = ' '.join(map(str, model['test']))
            out.write(f"{word} {word_vector_txt}\n")


if __name__ == '__main__':
    target_path = os.path.join(SETTINGS.VOCAB_PATH, 'target.txt')
    fasttext_model = os.path.join(SETTINGS.DATA_DIR, 'shrinked_fasttext.model')
    out_model = os.path.join(SETTINGS.DATA_DIR, 'target_vectors.txt')

    create_w2v_from_fasttext(
        fasttext_model_path=fasttext_model,
        vocab_path=target_path,
        w2v_out_path=out_model
    )
