import logging
import os
import sys
from typing import Tuple

import torch
from allennlp.data import Vocabulary
from allennlp.data.iterators import BasicIterator
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.modules import TextFieldEmbedder, TokenEmbedder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders.embedding import Embedding
from allennlp.training import util as training_util
from allennlp.training.checkpointer import Checkpointer
from torch import optim

from word_gan.gan.candidate_selectors.group_selector import GroupSelector
from word_gan.gan.dataset import TextDatasetReader
from word_gan.gan.discriminator import Discriminator
from word_gan.gan.fasttext_indexer import StaticFasttextTokenIndexer
from word_gan.gan.generator import Generator
from word_gan.gan.helpers.loaders import load_w2v, load_fasttext
from word_gan.gan.train_logger import WordGanLogger
from word_gan.gan.trainer import GanTrainer
from word_gan.settings import SETTINGS


def get_model(vocab, device) -> Tuple[Generator, Discriminator]:
    token_fasttext_path = os.path.join(SETTINGS.DATA_DIR, 'shrinked_fasttext.model')
    token_fasttext_params_path = os.path.join(SETTINGS.DATA_DIR, 'shrinked_fasttext.model.params')

    target_w2v_model_path = os.path.join(SETTINGS.DATA_DIR, 'target_vectors.txt')

    print('target size', vocab.get_vocab_size('target'))
    print('tokens size', vocab.get_vocab_size('tokens'))

    target_w2v_embedding: Embedding = load_w2v(
        weights_file=target_w2v_model_path,
        vocab=vocab,
        device=device,
        namespace='target'
    )

    on_disk = True

    token_fasttext_embedding: TokenEmbedder = load_fasttext(
        model_path=token_fasttext_path,
        model_params_path=token_fasttext_params_path,
        vocab=vocab,
        namespace='tokens',
        on_disk=on_disk
    )

    text_field_embedder = BasicTextFieldEmbedder(
        token_embedders={
            "tokens-ngram": token_fasttext_embedding
        },
        embedder_to_indexer_map={
            "tokens-ngram": [
                "tokens-ngram",
                "tokens-ngram-lengths",
                "tokens-ngram-mask"
            ] if not on_disk else ['tokens']
        },
        allow_unmatched_keys=True
    )

    candidates_selector = GroupSelector(
        vocab=vocab,
        target_w2v=target_w2v_embedding,
        groups_file=SETTINGS.CANDIDATE_GROUPS_FILE,
        device=device
    )

    generator: Generator = Generator(
        text_embedder=text_field_embedder,
        vocab=vocab,
        candidates_selector=candidates_selector,
        generator_context_size=SETTINGS.GENERATOR_CONTEXT,
        discriminator_context_size=SETTINGS.MAX_CONTEXT_SIZE
    )

    discriminator: Discriminator = Discriminator(
        text_embedder=text_field_embedder,
        vocab=vocab,
        noise_std=target_w2v_embedding.weight.std() * 0.05,
        context_size=SETTINGS.MAX_CONTEXT_SIZE
    )

    return generator, discriminator


def launch_train(text_data_path):
    if torch.cuda.is_available():
        cuda_device = 0
    else:
        cuda_device = None

    vocab = Vocabulary.from_files(SETTINGS.VOCAB_PATH)

    synonym_words_path = os.path.join(SETTINGS.VOCAB_PATH, 'target.txt')

    fasttext_indexer = StaticFasttextTokenIndexer(
        model_path=os.path.join(SETTINGS.DATA_DIR, 'shrinked_fasttext.model'),
        namespace='tokens',
        lowercase_tokens=True
    )

    reader = TextDatasetReader(
        dict_path=synonym_words_path,
        limit_words=-1,
        limit_freq=0,
        max_context_size=SETTINGS.MAX_CONTEXT_SIZE,
        token_indexers={
            "tokens": fasttext_indexer
        },
        target_indexers={
            "tokens": fasttext_indexer,
            "target": SingleIdTokenIndexer(
                namespace='target',
                lowercase_tokens=True
            )
        },
    )

    train_dataset = reader.read(text_data_path)

    iterator = BasicIterator(batch_size=SETTINGS.BATCH_SIZE)

    iterator.index_with(vocab)

    models: Tuple[Generator, Discriminator] = get_model(vocab, device=cuda_device)

    generator, discriminator = models

    if cuda_device is not None:
        generator = generator.cuda(cuda_device)
        discriminator = discriminator.cuda(cuda_device)

    generator_optimizer = optim.Adam(generator.parameters(), lr=0.001)
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=0.001)

    training_util.move_optimizer_to_cuda(generator_optimizer)
    training_util.move_optimizer_to_cuda(discriminator_optimizer)

    serialization_dir = os.path.join(SETTINGS.DATA_DIR, 'serialization')

    generator_checkpoint_path = os.path.join(serialization_dir, 'generator')
    os.makedirs(generator_checkpoint_path, exist_ok=True)
    generator_checkpointer = Checkpointer(
        serialization_dir=generator_checkpoint_path,
        num_serialized_models_to_keep=1
    )

    discriminator_checkpoint_path = os.path.join(serialization_dir, 'discriminator')
    os.makedirs(discriminator_checkpoint_path, exist_ok=True)
    discriminator_checkpointer = Checkpointer(
        serialization_dir=discriminator_checkpoint_path,
        num_serialized_models_to_keep=1
    )

    logger = WordGanLogger(
        serialization_path=os.path.join(serialization_dir, 'train_examples.txt'),
        batch_period=99,
        vocab=vocab
    )

    trainer = GanTrainer(
        serialization_dir=serialization_dir,
        data=train_dataset,
        generator=generator,
        discriminator=discriminator,
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        generator_checkpointer=generator_checkpointer,
        discriminator_checkpointer=discriminator_checkpointer,
        batch_iterator=iterator,
        cuda_device=cuda_device,
        max_batches=50,
        num_epochs=int(os.getenv("EPOCHS", 2)),
        train_logger=logger
    )

    trainer.train()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) > 1:
        text_data_path = sys.argv[1]
    else:
        text_data_path = os.path.join(SETTINGS.DATA_DIR, 'train_data.txt')

    launch_train(text_data_path)
