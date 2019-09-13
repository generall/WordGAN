from typing import Dict

import torch
from allennlp.modules import FeedForward
from allennlp.nn import Activation
from torch import nn

from word_gan.model.multilayer_cnn import MultilayerCnnEncoder


class SynonymDiscriminator(nn.Module):

    def __init__(self, embedding_dim):
        super(SynonymDiscriminator, self).__init__()

        self.embedding_dim = embedding_dim
        self.encoder = MultilayerCnnEncoder(
            embedding_dim=self.embedding_dim,
            num_filters=self.embedding_dim,
            layers=2,
            ngram_filter_sizes=(2, 3),
            output_dim=self.embedding_dim
        )

        self.ff = FeedForward(
            input_dim=self.embedding_dim * 2,
            num_layers=2,
            hidden_dims=[self.embedding_dim, 1],
            activations=[Activation.by_name('relu')(),
                         Activation.by_name('linear')()],
            dropout=[0.2, 0]
        )

    def forward(self, left_context, word, right_context) -> Dict[str, torch.Tensor]:
        """

        :param left_context: [batch_size, context_size, embedding_size]
        :param word: vector of the word of interest [batch_size, embedding_size]
        :param right_context: [batch_size, context_size, embedding_size]
        :param labels: if the `word` is original or not. Shape: [batch_size]
        :return:
        """

        # shape: [batch_size, sent_length, embedding_size]
        sentence_tensor = torch.cat([
            left_context,
            word.unsqueeze(1),
            right_context
        ], dim=1)

        # shape: [batch_size, embedding_size]
        encoded_sentence = self.encoder(sentence_tensor)

        # shape: [batch_size, embedding_size * 2]
        ff_input = torch.cat([encoded_sentence, word], dim=1)

        # shape: [batch_size]
        result_scores = self.ff(ff_input).squeeze()

        return result_scores
