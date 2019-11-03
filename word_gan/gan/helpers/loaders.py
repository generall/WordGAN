import os

import h5py
from allennlp.modules import Embedding, TokenEmbedder
from allennlp.modules.token_embedders.embedding import _read_pretrained_embeddings_file, _read_embeddings_from_hdf5
from loguru import logger

from word_gan.model.embedding_bag import DiskGensimTokenEmbedder, FastTextTokenEmbedder
from word_gan.model.thrifty_embeddings import ThriftyEmbedding
from word_gan.settings import SETTINGS


def load_w2v(
        weights_file,
        vocab,
        namespace='tokens',
        device=None
) -> Embedding:
    cache_file = weights_file + '.cache.hd5'

    if os.path.exists(cache_file):
        weights = _read_embeddings_from_hdf5(cache_file,
                                             embedding_dim=SETTINGS.EMBEDDINGS_SIZE,
                                             vocab=vocab,
                                             namespace=namespace)
    else:
        weights = _read_pretrained_embeddings_file(
            weights_file,
            SETTINGS.EMBEDDINGS_SIZE,
            vocab,
            namespace=namespace
        )

        with h5py.File(cache_file, 'w') as f:
            f.create_dataset("embedding", data=weights.numpy())

    if device is not None:
        weights = weights.cuda(device)

    logger.info(f"W2V size: {weights.shape}")

    token_embedding = ThriftyEmbedding(
        trainable=False,
        weights_file=weights_file,
        num_embeddings=vocab.get_vocab_size(namespace),
        weight=weights,
        embedding_dim=SETTINGS.EMBEDDINGS_SIZE
    )

    return token_embedding


def load_fasttext(model_path, model_params_path, vocab, namespace, on_disk=True) -> TokenEmbedder:
    if on_disk:
        embedder = DiskGensimTokenEmbedder(
            vocab=vocab,
            model_path=model_path,
            model_params_path=model_params_path,
            vocab_namespace=namespace,
            trainable=False
        )
    else:
        embedder = FastTextTokenEmbedder(model_path=model_path, trainable=False)

    return embedder
