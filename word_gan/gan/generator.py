from typing import Dict

import torch
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Embedding
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.training.metrics import Average, BooleanAccuracy
from torch import nn
from torch.nn import Linear

from word_gan.gan.candidate_selectors.base_selector import CandidatesSelector
from word_gan.model.selection_generator import SelectionGenerator
from word_gan.model.synonym_discriminator import SynonymDiscriminator


class Generator(Model):

    def __init__(
            self,
            text_embedder: TextFieldEmbedder,
            vocab: Vocabulary,
            candidates_selector: CandidatesSelector,
            generator_context_size=2,
            discriminator_context_size=4,
            additional_embedding_dim=20
    ):
        """

        :param vocab:
        """
        super().__init__(vocab)

        self.target_additional_embedding = Embedding(
            num_embeddings=vocab.get_vocab_size('target'),
            embedding_dim=additional_embedding_dim,
            vocab_namespace='target'
        )

        self.additional_embedder = BasicTextFieldEmbedder(
            token_embedders={
                'target': self.target_additional_embedding
            },
            embedder_to_indexer_map={'target': ['target']},
            allow_unmatched_keys=True
        )

        self.projection = Linear(
            in_features=text_embedder.get_output_dim() + additional_embedding_dim,
            out_features=text_embedder.get_output_dim()
        )

        self.candidates_selector = candidates_selector
        self.text_embedder = text_embedder

        self.selection_generator = SelectionGenerator(text_embedder.get_output_dim())

        self.generator_context_size = generator_context_size
        self.discriminator_context_size = discriminator_context_size

        self.loss = nn.BCEWithLogitsLoss()

        self.accuracy = BooleanAccuracy()

        self.attention_sharpness = Average()

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            'accuracy': self.accuracy.get_metric(reset),
            'attention_sharpness': self.attention_sharpness.get_metric(reset)
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
        left_context_vectors = self.text_embedder(left_context)
        right_context_vectors = self.text_embedder(right_context)

        # shape: [batch_size, embedding_dim]
        word_vectors = self.text_embedder(word).squeeze(1)

        # shape: [batch_size, additional_embedding_dim]
        additional_vectors = self.additional_embedder(word).squeeze(1)

        # shape: [batch_size, embedding_dim]
        word_vectors = self.projection(torch.cat([word_vectors, additional_vectors], dim=1))

        generator_left_context = left_context_vectors[:, -self.generator_context_size:, :]
        generator_right_context = right_context_vectors[:, :self.generator_context_size, :]

        # target indexes, variant vectors, variant masks
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

        attention_max = synonym_words_score.max(dim=1)[0].mean()
        self.attention_sharpness(attention_max)

        # [batch_size]
        selected_variant_indexes = torch.argmax(synonym_words_score, dim=1)
        target_synonym_indexes = variant_ids.gather(dim=1, index=selected_variant_indexes.unsqueeze(-1)).squeeze()

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

            discriminator_probs = torch.sigmoid(discriminator_predictions)

            # We want to trick discriminator here
            required_predictions = torch.ones_like(discriminator_predictions)

            discriminator_vals = (discriminator_probs > 0.5).long()
            self.accuracy(discriminator_vals, required_predictions.long())

            guess_loss = self.loss(discriminator_predictions, required_predictions)

            result['discriminator_predictions'] = discriminator_predictions
            result['loss'] = guess_loss

        return result
