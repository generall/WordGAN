import os
import sys
from typing import Union, List, Dict, Any, Iterable

import torch
from allennlp.data import Vocabulary, DataIterator, Instance
from allennlp.data.iterators import BucketIterator, BasicIterator
from allennlp.models import Model
from allennlp.training import TrainerBase
from torch.optim.optimizer import Optimizer

from word_gan.gan.dataset import TextDatasetReader
from word_gan.model.discriminator import Discriminator
from word_gan.model.generator import Generator
from word_gan.settings import DATA_DIR, SETTINGS

BATCH_SIZE = 64


def load_vocab(target_vocab_path, tokens_vocab_path):
    vocab = Vocabulary()

    vocab.set_from_file(
        filename=target_vocab_path,
        is_padded=False,
        namespace='target'
    )

    vocab.set_from_file(
        filename=tokens_vocab_path,
        is_padded=True,
        namespace='tokens'
    )

    return vocab


class GanTrainer(TrainerBase):

    def __init__(self,
                 serialization_dir: str,
                 data: Iterable[Instance],
                 generator: Generator,
                 discriminator: Discriminator,
                 generator_optimizer: Optimizer,
                 discriminator_optimizer: Optimizer,
                 batch_iterator: DataIterator,
                 cuda_device: Union[int, List] = -1,
                 max_batches: int = 100,
                 num_epochs: int = 10
                 ) -> None:
        """

        :param serialization_dir:
        :param batch_iterator:
        :param cuda_device:
        :param max_batches: max batches per epoch
        """
        super().__init__(serialization_dir, cuda_device)

        self.discriminator_optimizer = discriminator_optimizer
        self.generator_optimizer = generator_optimizer

        self.discriminator: Discriminator = discriminator
        self.generator: Generator = generator

        self.data = data
        self.batch_iterator = batch_iterator
        self.max_batches = max_batches
        self.num_epochs = num_epochs

    def train_one_epoch(self) -> Dict[str, float]:
        self.generator.train()
        self.discriminator.train()

        generator_loss = 0.0
        discriminator_real_loss = 0.0
        discriminator_fake_loss = 0.0

        # First train the discriminator
        data_iterator = self.batch_iterator(self.data)

        discriminator_quota = self.max_batches
        generator_quota = self.max_batches

        for batch in data_iterator:
            # Train discriminator while it has quotas.
            # When it's empty, assign quotas to generator

            if discriminator_quota > 0:
                self.discriminator_optimizer.zero_grad()

                # Real example, want discriminator to predict 1.

                real_error = self.discriminator(**batch, label=torch.ones(BATCH_SIZE))["loss"]
                real_error.backward()

                # Fake example, want discriminator to predict 0.
                fake_data = self.generator(**batch)["output_indexes"]

                fake_batch = {
                    **batch,
                    'word': fake_data
                }

                fake_error = self.discriminator(**fake_batch, label=torch.zeros(BATCH_SIZE))["loss"]
                fake_error.backward()

                discriminator_real_loss += real_error.sum().item()
                discriminator_fake_loss += fake_error.sum().item()

                self.discriminator_optimizer.step()
                discriminator_quota -= 1

            elif generator_quota > 0:
                # Now train the generator
                self.generator_optimizer.zero_grad()

                generated = self.generator(**batch, discriminator=self.discriminator)
                fake_error = generated["loss"]
                fake_error.backward()

                generator_loss += fake_error.sum().item()

                self.generator_optimizer.step()
                generator_quota -= 1

            else:
                discriminator_quota = self.max_batches
                generator_quota = self.max_batches

        return {
            "generator_loss": generator_loss,
            "discriminator_fake_loss": discriminator_fake_loss,
            "discriminator_real_loss": discriminator_real_loss,
        }

    def train(self) -> Dict[str, Any]:
        pass  # ToDo


if __name__ == '__main__':

    freq_dict_path = os.path.join(DATA_DIR, 'count_1w.txt')

    if len(sys.argv) > 1:
        text_data_path = sys.argv[1]
    else:
        text_data_path = os.path.join(DATA_DIR, 'test_data.txt')

    vocab = load_vocab(SETTINGS.TARGET_VOCAB_PATH, SETTINGS.TOKEN_VOCAB_PATH)

    print('target size', vocab.get_vocab_size('target'))
    print('tokens size', vocab.get_vocab_size('tokens'))

    reader = TextDatasetReader(
        dict_path=freq_dict_path,
        limit_words=100_000,
        limit_freq=0,
        small_context=1,
        large_context=2
    )

    train_dataset = reader.read(text_data_path)

    iterator = BasicIterator(batch_size=BATCH_SIZE)
