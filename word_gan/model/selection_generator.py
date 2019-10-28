from typing import Optional, Tuple

import torch
from allennlp.modules import FeedForward, Attention, Seq2VecEncoder
from allennlp.modules.attention import LinearAttention, BilinearAttention
from allennlp.nn import Activation
from allennlp.nn.util import combine_tensors_and_multiply, weighted_sum
from torch import nn

from word_gan.model.attentions import MultilayerAttention, LinearDotAttention
from word_gan.model.multilayer_cnn import MultilayerCnnEncoder


class BaseSelectionGenerator(nn.Module):

    def __init__(
            self,
            encoder: Seq2VecEncoder,
            attention: Attention
    ):
        super(BaseSelectionGenerator, self).__init__()

        self.encoder = encoder
        self.attention = attention

    def forward(
            self,
            left_context: torch.Tensor,
            word: torch.Tensor,
            right_context: torch.Tensor,
            variants: torch.Tensor,
            variants_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Should select appropriate synonym vector from variants

        :param left_context: [batch_size, context_size, embedding_size]
        :param word: vector of the word of interest [batch_size, embedding_size]
        :param right_context: [batch_size, context_size, embedding_size]
        :param variants: [batch_size, variants_count, embedding_size] - possible vectors to select from
        :param variants_mask: [batch_size, variants_count]
        :return: synonyms vectors [batch_size, embedding_size] + attention
        """
        # Shape: [batch_size, sent_length, embedding_size]
        sentence_tensor = torch.cat([
            left_context,
            word.unsqueeze(1),
            right_context
        ], dim=1)

        # Shape: [batch_size, embedding_size]
        encoded_sentence = self.encoder(sentence_tensor)

        # Shape: [batch_size, embedding_size * 2]
        attention_keys = torch.cat([encoded_sentence, word], dim=1)

        # Shape: [batch_size, variants_count]
        attention_output = self.attention(
            vector=attention_keys,
            matrix=variants,
            matrix_mask=variants_mask
        )

        # Shape: [batch_size, embedding_size]
        selected_variants = weighted_sum(variants, attention_output)

        return selected_variants, attention_output


class SelectionGenerator(BaseSelectionGenerator):

    """
    This is the only thing which is actually work on synthetic data
    """

    def __init__(self, embedding_dim):
        self.embedding_dim = embedding_dim
        encoder = MultilayerCnnEncoder(
            embedding_dim=self.embedding_dim,
            num_filters=self.embedding_dim * 2,
            layers=2,
            conv_layer_activation=Activation.by_name('tanh')(),
            ngram_filter_sizes=(3,),
            output_dim=self.embedding_dim,
            pooling='avg'
        )

        attention = MultilayerAttention(
            vector_dim=self.embedding_dim * 2,
            matrix_dim=self.embedding_dim,
            hidden_dims=[self.embedding_dim],
            activations=[Activation.by_name('tanh')()],
            dropout=[0.0],
            normalize=True
        )

        super(SelectionGenerator, self).__init__(
            encoder=encoder,
            attention=attention,
        )


class DotSelectionGenerator(BaseSelectionGenerator):

    def __init__(self, embedding_dim):
        self.embedding_dim = embedding_dim
        encoder = MultilayerCnnEncoder(
            embedding_dim=self.embedding_dim,
            num_filters=self.embedding_dim,
            layers=2,
            conv_layer_activation=Activation.by_name('tanh')(),
            ngram_filter_sizes=(3,),
            output_dim=self.embedding_dim,
            pooling='avg'
        )

        attention = LinearDotAttention(
            vector_dim=self.embedding_dim * 2,
            matrix_dim=self.embedding_dim,
            normalize=True
        )

        super(DotSelectionGenerator, self).__init__(
            encoder=encoder,
            attention=attention,
        )


class BiLinearSelectionGenerator(BaseSelectionGenerator):

    def __init__(self, embedding_dim):
        self.embedding_dim = embedding_dim
        encoder = MultilayerCnnEncoder(
            embedding_dim=self.embedding_dim,
            num_filters=self.embedding_dim * 2,
            layers=2,
            conv_layer_activation=Activation.by_name('tanh')(),
            ngram_filter_sizes=(3,),
            output_dim=self.embedding_dim,
            pooling='avg'
        )

        attention = BilinearAttention(
            vector_dim=self.embedding_dim * 2,
            matrix_dim=self.embedding_dim,
            normalize=True
        )

        super(BiLinearSelectionGenerator, self).__init__(
            encoder=encoder,
            attention=attention,
        )
