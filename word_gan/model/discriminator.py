from typing import Dict, Optional

import torch
from allennlp.modules import TextFieldEmbedder
from allennlp.training.metrics import BooleanAccuracy
from torch import nn
from allennlp.data import Vocabulary
from allennlp.models import Model

from word_gan.model.synonym_discriminator import SynonymDiscriminator


class Discriminator(Model):
    context_size = 2

    def __init__(self,
                 w2v: TextFieldEmbedder,
                 vocab: Vocabulary):
        super().__init__(vocab)

        self.w2v = w2v
        self.synonym_discriminator = SynonymDiscriminator(w2v.get_output_dim())
        self.loss = nn.BCEWithLogitsLoss()

        self.accuracy = BooleanAccuracy()

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            'accuracy': self.accuracy.get_metric(reset)
        }

    def discriminate(self, left_context, word, right_context):
        """
        Discriminate word vectors

        :param left_context: [batch_size, context_size, embedding_size]
        :param word: vector of the word of interest [batch_size, embedding_size]
        :param right_context: [batch_size, context_size, embedding_size]
        :return: [batch_size]
        """
        return self.synonym_discriminator(
            left_context,
            word,
            right_context
        )

    def forward(
            self,
            left_context: Dict[str, torch.LongTensor],
            word: Dict[str, torch.LongTensor],
            right_context: Dict[str, torch.LongTensor],
            word_vectors: Optional[torch.LongTensor] = None,
            labels=None
    ) -> Dict[str, torch.Tensor]:
        """

        :param left_context:  Dict[str, torch.LongTensor]
        :param word:  Dict[str, torch.LongTensor]
        :param right_context:  Dict[str, torch.LongTensor]
        :param word_vectors: Discriminator can also use directly provided word vectors. Shape: [batch_size, embedding_dim]
        :param labels:
        :return:
        """

        # shape: [batch_size, context_size, embedding_size]
        left_context_vectors = self.w2v(left_context)
        right_context_vectors = self.w2v(right_context)

        if word_vectors is None:
            # shape: [batch_size, embedding_dim]
            word_vectors: torch.Tensor = self.w2v(word).squeeze(1)

        batch_size, *_ = word_vectors.shape

        discriminator_left_context = left_context_vectors[:, -self.context_size:, :]
        discriminator_right_context = right_context_vectors[:, :self.context_size, :]

        # shape: [batch_size]
        result_scores = self.discriminate(
            discriminator_left_context,
            word_vectors,
            discriminator_right_context
        )

        result_probs = torch.sigmoid(result_scores)

        result = {'probs': result_probs}

        if labels is not None:
            labels = word_vectors.new_full(size=(batch_size,), fill_value=labels)
            result_vals = (result_probs > 0.5).long()
            self.accuracy(result_vals, labels.long())
            result['loss'] = self.loss(result_scores, labels)
        else:
            result['probs'] = result_probs

        return result
