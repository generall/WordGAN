from typing import Union, List, Dict, Any, Iterable, Optional

import tqdm
from allennlp.data import DataIterator, Instance
from allennlp.training import TrainerBase
from allennlp.training.checkpointer import Checkpointer
from allennlp.training.metrics import BooleanAccuracy
from allennlp.training.tensorboard_writer import TensorboardWriter
from loguru import logger
from torch.optim.optimizer import Optimizer

from word_gan.gan.train_logger import TrainLogger
from word_gan.gan.discriminator import Discriminator
from word_gan.gan.generator import Generator
from allennlp.nn import util


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
                 generator_checkpointer: Checkpointer,
                 discriminator_checkpointer: Checkpointer,
                 batch_iterator: DataIterator,
                 cuda_device: Union[int, List] = -1,
                 max_batches: int = 5,  # ToDo: fix this
                 num_epochs: int = 10,
                 train_logger: Optional[TrainLogger] = None,
                 ) -> None:
        """

        :param serialization_dir:
        :param batch_iterator:
        :param cuda_device:
        :param max_batches: max batches per epoch
        """
        super().__init__(serialization_dir, cuda_device if cuda_device is not None else -1)

        self.discriminator_checkpointer = discriminator_checkpointer
        self.generator_checkpointer = generator_checkpointer

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

        self._batch_num_total = 0

        self._tensorboard = TensorboardWriter(
            get_batch_num_total=lambda: self._batch_num_total,
            serialization_dir=serialization_dir,
            summary_interval=5,
            histogram_interval=None,
            should_log_parameter_statistics=True,
            should_log_learning_rate=False
        )

    def _restore_checkpoint(self) -> int:
        generator_state, generator_training_state = self.generator_checkpointer.restore_checkpoint()

        if not generator_training_state:
            # No checkpoint to restore, start at 0
            return 0

        self.generator.load_state_dict(generator_state)
        self.generator_optimizer.load_state_dict(generator_training_state["optimizer"])

        batch_num_total = generator_training_state.get('batch_num_total')
        if batch_num_total is not None:
            self._batch_num_total = batch_num_total

        discriminator_state, discriminator_training_state = self.discriminator_checkpointer.restore_checkpoint()

        if not discriminator_training_state:
            # No checkpoint to restore, start at 0
            return 0

        self.discriminator.load_state_dict(discriminator_state)
        self.discriminator_optimizer.load_state_dict(discriminator_training_state["optimizer"])

        return generator_training_state.get('epoch', 0)

    def _save_checkpoint(self, epoch):
        generator_training_states = {
            "optimizer": self.generator_optimizer.state_dict(),
            "batch_num_total": self._batch_num_total
        }

        self.generator_checkpointer.save_checkpoint(
            epoch=epoch,
            model_state=self.generator.state_dict(),
            training_states=generator_training_states,
            is_best_so_far=False
        )

        discriminator_training_states = {
            "optimizer": self.discriminator_optimizer.state_dict(),
            "batch_num_total": self._batch_num_total
        }

        self.discriminator_checkpointer.save_checkpoint(
            epoch=epoch,
            model_state=self.discriminator.state_dict(),
            training_states=discriminator_training_states,
            is_best_so_far=False
        )

    def train_one_epoch(self) -> Dict[str, float]:
        self.generator.train()
        self.discriminator.train()

        generator_loss = 0.0
        discriminator_real_loss = 0.0
        discriminator_fake_loss = 0.0

        # First train the discriminator
        data_iterator = self.batch_iterator(self.data, num_epochs=1, shuffle=False)

        discriminator_quota = self.max_batches
        generator_quota = self.max_batches * 2

        for batch in tqdm.tqdm(data_iterator):
            # Train discriminator while it has quotas.
            # When it's empty, assign quotas to generator

            batch = util.move_to_device(batch, self._cuda_devices[0])

            self._batch_num_total += 1
            if discriminator_quota > 0:
                self.discriminator_optimizer.zero_grad()

                # Real example, want discriminator to predict 1.
                real_predictions = self.discriminator(**batch, labels=1)
                real_error = real_predictions["loss"]
                real_error.backward()

                # Fake example, want discriminator to predict 0.
                generator_output = self.generator(**batch)
                fake_data = generator_output["discriminator_overrides"]

                fake_batch = {
                    **batch,
                    **fake_data
                }

                fake_predictions = self.discriminator(**fake_batch, labels=0)
                fake_error = fake_predictions["loss"]
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
                    self.train_logger.log_info(batch, generated, self._batch_num_total)

                generator_quota -= 1

            else:
                discriminator_quota = self.max_batches
                generator_quota = self.max_batches

                discriminator_metrics = self.discriminator.get_metrics(reset=False)
                generator_metrics = self.generator.get_metrics(reset=False)

                metrics = {
                    "batch_generator_loss": generator_loss,
                    "batch_discriminator_fake_loss": discriminator_fake_loss,
                    "batch_discriminator_real_loss": discriminator_real_loss,
                    **add_prefix(discriminator_metrics, 'batch_discriminator'),
                    **add_prefix(generator_metrics, 'batch_generator'),
                }

                if generator_loss > discriminator_fake_loss:
                    generator_quota *= max(generator_loss / discriminator_fake_loss, 10)
                else:
                    discriminator_quota *= max(discriminator_fake_loss / generator_loss, 10)

                generator_loss = 0.0
                discriminator_real_loss = 0.0
                discriminator_fake_loss = 0.0

                self._tensorboard.log_metrics(train_metrics=metrics, epoch=None, log_to_console=False)

        discriminator_metrics = self.discriminator.get_metrics(reset=True)
        generator_metrics = self.generator.get_metrics(reset=True)
        return {
            # "generator_loss": generator_loss,
            # "discriminator_fake_loss": discriminator_fake_loss,
            # "discriminator_real_loss": discriminator_real_loss,
            **add_prefix(discriminator_metrics, 'discriminator'),
            **add_prefix(generator_metrics, 'generator'),
        }

    def train(self) -> Dict[str, Any]:

        restored_epoch = self._restore_checkpoint()
        logger.info(f"Restoring epoch: {restored_epoch}")

        metrics = {}
        for epoch in range(self.num_epochs):
            metrics = self.train_one_epoch()
            self._tensorboard.log_metrics(train_metrics=metrics, epoch=epoch, log_to_console=True)
            self._save_checkpoint(epoch)

        return metrics
