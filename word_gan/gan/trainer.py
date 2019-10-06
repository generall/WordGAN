from typing import Union, List, Dict, Any, Iterable, Optional

import numpy as np
import torch
import json
import tqdm
from allennlp.data import DataIterator, Instance
from allennlp.training import TrainerBase
from allennlp.training.metrics import BooleanAccuracy
from loguru import logger
from torch.optim.optimizer import Optimizer

from word_gan.gan.train_logger import TrainLogger
from word_gan.model.discriminator import Discriminator
from word_gan.model.generator import Generator


def add_prefix(dct: dict, prefix) -> dict:
    return dict((f"{prefix}_{key}", val) for key, val in dct.items())


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
                 num_epochs: int = 10,
                 train_logger: Optional[TrainLogger] = None,
                 ) -> None:
        """

        :param serialization_dir:
        :param batch_iterator:
        :param cuda_device:
        :param max_batches: max batches per epoch
        """
        super().__init__(serialization_dir, cuda_device)

        self.train_logger = train_logger
        self.discriminator_optimizer = discriminator_optimizer
        self.generator_optimizer = generator_optimizer

        self.discriminator: Discriminator = discriminator
        self.generator: Generator = generator

        self.data = data
        self.batch_iterator = batch_iterator
        self.max_batches = max_batches
        self.num_epochs = num_epochs

        self.discriminator_true_acc = BooleanAccuracy()
        self.discriminator_false_acc = BooleanAccuracy()
        self.generator_acc = BooleanAccuracy()

    def train_one_epoch(self) -> Dict[str, float]:
        self.generator.train()
        self.discriminator.train()

        generator_loss = 0.0
        discriminator_real_loss = 0.0
        discriminator_fake_loss = 0.0

        # First train the discriminator
        data_iterator = self.batch_iterator(self.data, num_epochs=1)

        discriminator_quota = self.max_batches
        generator_quota = self.max_batches

        for batch in tqdm.tqdm(data_iterator):
            # Train discriminator while it has quotas.
            # When it's empty, assign quotas to generator

            if discriminator_quota > 0:
                self.discriminator_optimizer.zero_grad()

                # Real example, want discriminator to predict 1.
                real_predictions = self.discriminator(**batch, labels=1)
                real_error = real_predictions["loss"]
                real_error.backward()

                # Fake example, want discriminator to predict 0.
                fake_data = self.generator(**batch)["output_indexes"]

                fake_batch = {
                    **batch,
                    'word': fake_data
                }

                fake_error = self.discriminator(**fake_batch, labels=0)["loss"]
                fake_error.backward()

                discriminator_real_loss += real_error.sum().item()
                discriminator_fake_loss += fake_error.sum().item()

                self.discriminator_optimizer.step()
                discriminator_quota -= 1

            elif generator_quota > 0:
                # Now train the generator
                self.generator_optimizer.zero_grad()

                generated = self.generator(**batch, discriminator=self.discriminator.synonym_discriminator)
                fake_error = generated["loss"]
                fake_error.backward()

                generator_loss += fake_error.sum().item()

                self.generator_optimizer.step()

                if self.train_logger:
                    self.train_logger.log_generator(batch, generated)

                generator_quota -= 1

            else:
                discriminator_quota = self.max_batches
                generator_quota = self.max_batches

        discriminator_metrics = self.discriminator.get_metrics(reset=True)
        generator_metrics = self.generator.get_metrics(reset=True)

        return {
            "generator_loss": generator_loss,
            "discriminator_fake_loss": discriminator_fake_loss,
            "discriminator_real_loss": discriminator_real_loss,
            **add_prefix(discriminator_metrics, 'discriminator'),
            **add_prefix(generator_metrics, 'generator'),
        }

    def train(self) -> Dict[str, Any]:
        metrics = {}
        for _ in range(self.num_epochs):
            metrics = self.train_one_epoch()

            logger.info(f"Metrics: {json.dumps(metrics, indent=2)}")

        return metrics
