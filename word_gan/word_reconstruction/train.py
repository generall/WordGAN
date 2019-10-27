"""
Script for training model which restores word from vector in differentiable way
"""
import logging
import os
import sys

import torch
from allennlp.data import Vocabulary
from allennlp.modules.token_embedders.embedding import _read_pretrained_embeddings_file

from word_gan.model.embedding_to_word import EmbeddingToWord

from word_gan.word_reconstruction.dataset import DictDatasetReader

os.environ['MODE'] = 'synthetic'

from word_gan.settings import SETTINGS

NAMESPACE = 'target'
EMBEDDING_DIM = SETTINGS.EMBEDDINGS_SIZE
DEFAULT_EMBEDDING_PATH = os.path.join(SETTINGS.DATA_DIR, 'model.txt')
OUT_MODEL_PATH = os.getenv("OUT_MODEL_PATH", os.path.join(SETTINGS.DATA_DIR, 'v2w_model.th'))
OUT_VOCAB_PATH = os.getenv("OUT_VOCAB_PATH", os.path.join(SETTINGS.DATA_DIR, 'vocab'))
INIT_VOCAB_PATH = os.getenv('INIT_VOCAB_PATH', os.path.join(SETTINGS.DATA_DIR, 'common.txt'))

EMBEDDING_PATH = os.getenv('EMBEDDING_PATH', DEFAULT_EMBEDDING_PATH)
NUM_WORDS = int(os.getenv("NUM_WORDS", 100_000))


def build_v2w(vocab):
    print(f"vocab {NAMESPACE} size:", vocab.get_vocab_size(namespace=NAMESPACE))

    weights = _read_pretrained_embeddings_file(EMBEDDING_PATH, EMBEDDING_DIM, vocab, namespace=NAMESPACE)

    print('weights.shape:', weights.shape)

    v2w: EmbeddingToWord = EmbeddingToWord(embedding_size=EMBEDDING_DIM, words_count=vocab.get_vocab_size(NAMESPACE))

    v2w.init_from_embeddings(weights)

    torch.save(v2w.state_dict(), OUT_MODEL_PATH)


# WARN: Deprecated
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    reader = DictDatasetReader(limit_words=NUM_WORDS, namespace=NAMESPACE)
    train_dataset = reader.read(INIT_VOCAB_PATH)

    vocab = Vocabulary.from_instances(train_dataset, non_padded_namespaces=[])

    vocab.save_to_files(OUT_VOCAB_PATH)
