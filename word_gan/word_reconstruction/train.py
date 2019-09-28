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
from word_gan.settings import DATA_DIR, TEST_DATA_DIR
from word_gan.word_reconstruction.dataset import DictDatasetReader

EMBEDDING_DIM = 300
DEFAULT_EMBEDDING_PATH = os.path.join(TEST_DATA_DIR, 'model_small.txt')
OUT_MODEL_PATH = os.getenv("OUT_MODEL_PATH", os.path.join(TEST_DATA_DIR, 'test_v2w_model.th'))
NAMESPACE = 'target'

if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        data_path = os.path.join(DATA_DIR, 'count_1w.txt')
    else:
        data_path = sys.argv[1]

    num_words = int(os.getenv("NUM_WORDS", 10_000))

    EMBEDDING_PATH = os.getenv('EMBEDDING_PATH', DEFAULT_EMBEDDING_PATH)

    reader = DictDatasetReader(limit_words=num_words, namespace=NAMESPACE)
    train_dataset = reader.read(data_path)

    vocab = Vocabulary.from_instances(train_dataset)

    vocab.save_to_files(os.path.join(TEST_DATA_DIR, 'vocab'))

    weights = _read_pretrained_embeddings_file(EMBEDDING_PATH, EMBEDDING_DIM, vocab, namespace=NAMESPACE)

    print('weights.shape:', weights.shape)

    v2w: EmbeddingToWord = EmbeddingToWord(embedding_size=EMBEDDING_DIM, words_count=vocab.get_vocab_size(NAMESPACE))

    v2w.init_from_embeddings(weights)

    torch.save(v2w.state_dict(), OUT_MODEL_PATH)
