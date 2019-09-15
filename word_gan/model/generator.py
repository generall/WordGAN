from typing import Dict

import torch
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder
from torch import nn

from word_gan.model.discriminator import Discriminator
from word_gan.model.synonym_discriminator import SynonymDiscriminator
from word_gan.model.embedding_to_word import EmbeddingToWord
from word_gan.model.synonyms_generator import SynonymGenerator
from word_gan.model.word_reconstruction import WordReconstruction


class Generator(Model):

    def __init__(
            self,
            embedding_dim,
            w2v: TextFieldEmbedder,
            v2w: EmbeddingToWord,
            vocab: Vocabulary,
            synonym_delta: float = 0.1
    ):
        """

        :param embedding_dim:
        :param v2w: vector to word mapping layer. Not trainable
        :param vocab:
        """
        super().__init__(vocab)

        self.synonym_delta = synonym_delta
        self.w2v = w2v
        self.v2w = v2w
        self.synonyms_generator = SynonymGenerator(embedding_dim)

        self.generator_context_size = 1
        self.discriminator_context_size = Discriminator.context_size

        self.loss = nn.BCEWithLogitsLoss()

    def forward(
            self,
            left_context:  Dict[str, torch.LongTensor],
            word:  Dict[str, torch.LongTensor],
            right_context:  Dict[str, torch.LongTensor],
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

        # shape: [batch_size, embedding_size]
        synonym_vectors = self.synonyms_generator(generator_left_context, word_vectors, generator_right_context)

        # shape: [batch_size, vocab_size]
        synonym_words_score = self.v2w(synonym_vectors)

        result = {
            'synonym_words_score': synonym_words_score
        }

        if discriminator:
            discriminator_left_context = left_context_vectors[:, -self.discriminator_context_size:, :]
            discriminator_right_context = right_context_vectors[:, :self.discriminator_context_size, :]

            discriminator_predictions = discriminator(
                discriminator_left_context,
                synonym_vectors,
                discriminator_right_context
            )

            # [batch_size, 1]
            target_indexes = word['target'].unsqueeze(1)

            # shape: [batch_size, vocab_size]
            delta_probs = torch.zeros_like(synonym_words_score).scatter_(1, target_indexes, self.synonym_delta)

            # shape: [batch_size, vocab_size]
            adjusted_synonym_words_score = synonym_words_score + delta_probs

            # shape: [batch_size], [batch_size]
            max_synonym_scores, synonym_words_ids = torch.max(adjusted_synonym_words_score, dim=1)

            # shape: [batch_size]
            loss_mask = word['target'] == synonym_words_ids

            # shape: [wrond_synonym_batch]
            wrong_synonyms_scores = max_synonym_scores[~loss_mask]

            # shape: [true_synonym_batch, vocab_size]
            discriminator_predictions = discriminator_predictions[loss_mask]

            # We want to trick discriminator here
            required_predictions = torch.ones_like(discriminator_predictions)

            guess_loss = self.loss(discriminator_predictions, required_predictions)

            # If generated synonym is same as initial word - the loss is this synonym probability
            # If not - loss is obtained from ability to trick discriminator
            result['loss'] = guess_loss + torch.sum(wrong_synonyms_scores)

        return result

