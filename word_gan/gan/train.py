import os
import sys

import torch
from allennlp.data import Vocabulary
from allennlp.data.iterators import BasicIterator
from allennlp.modules import TextFieldEmbedder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders.embedding import _read_pretrained_embeddings_file

from word_gan.gan.dataset import TextDatasetReader
from word_gan.gan.trainer import BATCH_SIZE
from word_gan.model.discriminator import Discriminator
from word_gan.model.embedding_to_word import EmbeddingToWord
from word_gan.model.generator import Generator
from word_gan.model.thrifty_embeddings import ThriftyEmbedding
from word_gan.settings import DATA_DIR, SETTINGS


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

    token_embedding = ThriftyEmbedding(
        trainable=False,
        weights_file=weights_file,
        num_embeddings=vocab.get_vocab_size(namespace),
        weight=weights,
        embedding_dim=SETTINGS.EMBEDDINGS_SIZE,
    )

    word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})

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

    freq_dict_path = os.path.join(DATA_DIR, 'common.txt')
    v2w_model_path = os.path.join(DATA_DIR, 'v2w_model.th')
    w2v_model_path = os.path.join(DATA_DIR, 'model.txt')

    if len(sys.argv) > 1:
        text_data_path = sys.argv[1]
    else:
        text_data_path = os.path.join(DATA_DIR, 'test_data.txt')

    vocab = Vocabulary.from_files(SETTINGS.VOCAB_PATH)

    print('target size', vocab.get_vocab_size('target'))
    print('tokens size', vocab.get_vocab_size('tokens'))

    reader = TextDatasetReader(
        dict_path=freq_dict_path,
        limit_words=vocab.get_vocab_size('target'),
        limit_freq=0
    )

    train_dataset = reader.read(text_data_path)

    iterator = BasicIterator(batch_size=BATCH_SIZE)

    iterator.index_with(vocab)

    v2w_model: EmbeddingToWord = load_v2w(
        weights_file=v2w_model_path,
        vocab=vocab
    )

    w2v_model: TextFieldEmbedder = load_w2v(
        weights_file=w2v_model_path,
        vocab=vocab,
    )

    generator = Generator(
        w2v=w2v_model,
        v2w=v2w_model,
        vocab=vocab
    )

    discriminator = Discriminator(
        w2v=w2v_model,
        vocab=vocab
    )
