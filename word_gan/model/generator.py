from typing import Dict, Tuple, Optional

import torch
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Attention
from allennlp.nn import Activation
from allennlp.training.metrics import Average, BooleanAccuracy
from torch import nn
from torch.nn.parameter import Parameter

from word_gan.model.attentions import MultilayerAttention
from word_gan.model.candidates_selector import CandidatesSelector
from word_gan.model.discriminator import Discriminator
from word_gan.model.selection_generator import SelectionGenerator
from word_gan.model.synonym_discriminator import SynonymDiscriminator


class Generator(Model):

    def __init__(
            self,
            w2v: TextFieldEmbedder,
            vocab: Vocabulary,
            candidates_selector: CandidatesSelector
    ):
        """

        :param vocab:
        """
        super().__init__(vocab)

        self.candidates_selector = candidates_selector
        self.w2v = w2v

        self.selection_generator = SelectionGenerator(w2v.get_output_dim())

        self.generator_context_size = 1
        self.discriminator_context_size = Discriminator.context_size

        self.loss = nn.BCEWithLogitsLoss()

        self.accuracy = BooleanAccuracy()

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            'accuracy': self.accuracy.get_metric(reset),
        }

    def forward(
            self,
            left_context: Dict[str, torch.LongTensor],
            word: Dict[str, torch.LongTensor],
            right_context: Dict[str, torch.LongTensor],
            discriminator: SynonymDiscriminator = None
    ) -> Dict[str, torch.Tensor]:
        """
        :param left_context: Dict[str, torch.LongTensor],
        :param word: Dict[str, torch.LongTensor],
        :param right_context: Dict[str, torch.LongTensor],
        :param discriminator:
        :return:
        """

        # shape: [batch_size, context_size, embedding_size]
        left_context_vectors = self.w2v(left_context)
        right_context_vectors = self.w2v(right_context)

        # shape: [batch_size, embedding_dim]
        word_vectors = self.w2v(word).squeeze(1)

        generator_left_context = left_context_vectors[:, -self.generator_context_size:, :]
        generator_right_context = right_context_vectors[:, :self.generator_context_size, :]

        # shape: [batch_size, num_variants], [batch_size, num_variants, emb_size], [batch_size, num_variants]
        variant_ids, variants, variants_mask = self.candidates_selector.get_candidates(word)

        # shape: [batch_size, embedding_size], [batch_size num_variants]
        synonym_vectors, synonym_words_score = self.selection_generator(
            generator_left_context,
            word_vectors,
            generator_right_context,
            variants=variants,
            variants_mask=variants_mask
        )

        # [batch_size]
        selected_variant_indexes = torch.argmax(synonym_words_score, dim=1)
        target_synonym_indexes = variant_ids[selected_variant_indexes]

        result = {
            'output_scores': synonym_words_score,
            'generated_indexes': {
                "target_indexes": target_synonym_indexes,
            },
            'discriminator_overrides': {
                "word_vectors": synonym_vectors
            }
        }

        if discriminator:
            discriminator_left_context = left_context_vectors[:, -self.discriminator_context_size:, :]
            discriminator_right_context = right_context_vectors[:, :self.discriminator_context_size, :]

            discriminator_predictions = discriminator(
                discriminator_left_context,
                synonym_vectors,
                discriminator_right_context
            )

            # We want to trick discriminator here
            required_predictions = torch.ones_like(discriminator_predictions)

            discriminator_vals = (discriminator_predictions > 0.5).long()
            self.accuracy(discriminator_vals, required_predictions.long())

            guess_loss = self.loss(discriminator_predictions, required_predictions)

            # If generated synonym is same as initial word - the loss is this synonym probability
            # If not - loss is obtained from ability to trick discriminator
            result['loss'] = guess_loss

        return result

