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
from word_gan.settings import DATA_DIR, TEST_DATA_DIR, SETTINGS
from word_gan.word_reconstruction.dataset import DictDatasetReader

EMBEDDING_DIM = SETTINGS.EMBEDDINGS_SIZE
DEFAULT_EMBEDDING_PATH = os.path.join(TEST_DATA_DIR, 'model_small.txt')
OUT_MODEL_PATH = os.getenv("OUT_MODEL_PATH", os.path.join(TEST_DATA_DIR, 'test_v2w_model.th'))
NAMESPACE = 'target'
OUT_VOCAB_PATH = os.getenv("OUT_VOCAB_PATH", os.path.join(TEST_DATA_DIR, 'vocab'))
NUM_WORDS = int(os.getenv("NUM_WORDS", 10_000))
EMBEDDING_PATH = os.getenv('EMBEDDING_PATH', DEFAULT_EMBEDDING_PATH)
INIT_VOCAB_PATH = os.getenv('INIT_VOCAB_PATH', os.path.join(DATA_DIR, 'count_1w.txt'))

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    reader = DictDatasetReader(limit_words=NUM_WORDS, namespace=NAMESPACE)
    train_dataset = reader.read(INIT_VOCAB_PATH)

    vocab = Vocabulary.from_instances(train_dataset)

    vocab.save_to_files(OUT_VOCAB_PATH)

    weights = _read_pretrained_embeddings_file(EMBEDDING_PATH, EMBEDDING_DIM, vocab, namespace=NAMESPACE)

    print('weights.shape:', weights.shape)

    v2w: EmbeddingToWord = EmbeddingToWord(embedding_size=EMBEDDING_DIM, words_count=vocab.get_vocab_size(NAMESPACE))

    v2w.init_from_embeddings(weights)

    torch.save(v2w.state_dict(), OUT_MODEL_PATH)
