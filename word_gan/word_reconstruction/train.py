"""
Script for training model which restores word from vector in differentiable way
"""
import os
import sys

import torch
from allennlp.data import Vocabulary
from allennlp.data.iterators import BucketIterator, BasicIterator
from allennlp.modules import Embedding
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders.embedding import _read_pretrained_embeddings_file
from allennlp.training import Trainer
from torch import optim

from word_gan.model.embedding_to_word import EmbeddingToWord
from word_gan.model.thrifty_embeddings import ThriftyEmbedding
from word_gan.model.word_reconstruction import WordReconstruction
from word_gan.settings import DATA_DIR
from word_gan.word_reconstruction.dataset import DictDatasetReader

EMBEDDING_DIM = 300
DEFAULT_EMBEDDING_PATH = os.path.join(DATA_DIR, 'model_small.txt')


def get_new_model(vocab, embedding_path):
    weights = _read_pretrained_embeddings_file(embedding_path, EMBEDDING_DIM, vocab)

    token_embedding = ThriftyEmbedding(
        trainable=False,
        weights_file=embedding_path,
        num_embeddings=vocab.get_vocab_size('tokens'),
        weight=weights,
        embedding_dim=EMBEDDING_DIM,
    )
    word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})

    embedding_to_word = EmbeddingToWord(EMBEDDING_DIM, vocab.get_vocab_size('tokens'))

    return WordReconstruction(
        w2v=word_embeddings,
        v2w=embedding_to_word,
        vocab=vocab
    )


if __name__ == '__main__':

    if len(sys.argv) < 2:
        data_path = os.path.join(DATA_DIR, 'count_1w.txt')
    else:
        data_path = sys.argv[1]

    num_words = int(os.getenv("NUM_WORDS", 10_000))

    EMBEDDING_PATH = os.getenv('EMBEDDING_PATH', DEFAULT_EMBEDDING_PATH)

    BATCH_SIZE = int(os.getenv("BATCH_SIZE", 64))

    NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", 200))

    reader = DictDatasetReader(limit_words=num_words)
    train_dataset = reader.read(data_path)

    vocab = Vocabulary.from_instances(train_dataset)

    model = get_new_model(vocab, EMBEDDING_PATH)

    if torch.cuda.is_available():
        cuda_device = 0
        model = model.cuda(cuda_device)
    else:
        cuda_device = -1

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    iterator = BasicIterator(batch_size=BATCH_SIZE)
    iterator.index_with(vocab)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        iterator=iterator,
        train_dataset=train_dataset,
        validation_dataset=train_dataset,
        num_epochs=NUM_EPOCHS,
        cuda_device=cuda_device,
        serialization_dir=os.path.join(DATA_DIR, 'serialization')
    )

    trainer.train()
