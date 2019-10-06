import logging
import os
import sys

import torch
from allennlp.data import Vocabulary
from allennlp.data.iterators import BasicIterator
from allennlp.modules import TextFieldEmbedder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders.embedding import _read_pretrained_embeddings_file
from loguru import logger
from torch import optim

from word_gan.gan.dataset import TextDatasetReader
from word_gan.gan.train_logger import WordGanLogger
from word_gan.gan.trainer import GanTrainer
from word_gan.model.discriminator import Discriminator
from word_gan.model.embedding_to_word import EmbeddingToWord
from word_gan.model.generator import Generator
from word_gan.model.thrifty_embeddings import ThriftyEmbedding
from word_gan.settings import SETTINGS


def load_w2v(
        weights_file,
        vocab,
        namespace='tokens',
):
    weights = _read_pretrained_embeddings_file(
        weights_file,
        SETTINGS.EMBEDDINGS_SIZE,
        vocab,
        namespace=namespace
    )

    logger.info(f"W2V size: {weights.shape}")

    token_embedding = ThriftyEmbedding(
        trainable=False,
        weights_file=weights_file,
        num_embeddings=vocab.get_vocab_size(namespace),
        weight=weights,
        embedding_dim=SETTINGS.EMBEDDINGS_SIZE
    )

    word_embeddings = BasicTextFieldEmbedder(
        {"tokens": token_embedding},
        allow_unmatched_keys=True
    )

    return word_embeddings


def load_v2w(
        weights_file,
        vocab,
        namespace='target',
        device=torch.device('cpu')
):
    model_state = torch.load(weights_file, map_location=device)

    model: torch.nn.Module = EmbeddingToWord(
        embedding_size=SETTINGS.EMBEDDINGS_SIZE,
        words_count=vocab.get_vocab_size(namespace)
    )

    model.load_state_dict(model_state)

    return model


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    freq_dict_path = os.path.join(SETTINGS.DATA_DIR, 'common.txt')
    v2w_model_path = os.path.join(SETTINGS.DATA_DIR, 'v2w_model.th')
    w2v_model_path = os.path.join(SETTINGS.DATA_DIR, 'model.txt')

    if len(sys.argv) > 1:
        text_data_path = sys.argv[1]
    else:
        text_data_path = os.path.join(SETTINGS.DATA_DIR, 'train_data.txt')

    vocab = Vocabulary.from_files(SETTINGS.VOCAB_PATH)

    print('target size', vocab.get_vocab_size('target'))
    print('tokens size', vocab.get_vocab_size('tokens'))

    reader = TextDatasetReader(
        dict_path=freq_dict_path,
        limit_words=vocab.get_vocab_size('target'),
        limit_freq=0
    )

    train_dataset = reader.read(text_data_path)

    iterator = BasicIterator(batch_size=SETTINGS.BATCH_SIZE)

    iterator.index_with(vocab)

    v2w_model: EmbeddingToWord = load_v2w(
        weights_file=v2w_model_path,
        vocab=vocab
    )

    w2v_model: TextFieldEmbedder = load_w2v(
        weights_file=w2v_model_path,
        vocab=vocab,
    )

    generator: Generator = Generator(
        w2v=w2v_model,
        v2w=v2w_model,
        vocab=vocab
    )

    discriminator: Discriminator = Discriminator(
        w2v=w2v_model,
        vocab=vocab
    )

    generator_optimizer = optim.Adam(generator.parameters(), lr=0.001)
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=0.001)

    if torch.cuda.is_available():
        cuda_device = 0
        generator = generator.cuda(cuda_device)
        discriminator = discriminator.cuda(cuda_device)
    else:
        cuda_device = -1

    trainer = GanTrainer(
        serialization_dir=os.path.join(SETTINGS.DATA_DIR, 'serialization'),
        data=train_dataset,
        generator=generator,
        discriminator=discriminator,
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        batch_iterator=iterator,
        cuda_device=cuda_device,
        max_batches=50,
        num_epochs=5,
        train_logger=WordGanLogger()
    )

    trainer.train()
