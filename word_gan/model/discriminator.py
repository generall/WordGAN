from typing import Dict

import torch
from allennlp.modules import TextFieldEmbedder
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

    def forward(
            self,
            left_context: Dict[str, torch.LongTensor],
            word: Dict[str, torch.LongTensor],
            right_context: Dict[str, torch.LongTensor],
            labels=None
    ) -> Dict[str, torch.Tensor]:
        """

        :param left_context:  Dict[str, torch.LongTensor]
        :param word:  Dict[str, torch.LongTensor]
        :param right_context:  Dict[str, torch.LongTensor]
        :param labels:
        :return:
        """

        # shape: [batch_size, context_size, embedding_size]
        left_context_vectors = self.w2v(left_context)
        right_context_vectors = self.w2v(right_context)

        # shape: [batch_size, embedding_dim]
        word_vectors = self.w2v(word).squeeze(1)

        discriminator_left_context = left_context_vectors[:, -self.context_size:, :]
        discriminator_right_context = right_context_vectors[:, :self.context_size, :]

        result_scores = self.synonym_discriminator(
            discriminator_left_context,
            word_vectors,
            discriminator_right_context
        )

        result = {}

        if labels is not None:
            result['loss'] = self.loss(result_scores, labels)
        else:
            result['scores'] = torch.sigmoid(result_scores)

        return result
