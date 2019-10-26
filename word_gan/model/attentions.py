from typing import List, Union

import torch
from allennlp.modules import FeedForward
from torch.nn import Linear
from torch.nn.parameter import Parameter

from allennlp.modules.attention.attention import Attention
from allennlp.nn import Activation


@Attention.register("multilayer-attention")
class MultilayerAttention(Attention):
    def __init__(self,
                 vector_dim: int,
                 matrix_dim: int,
                 hidden_dims: List[int],
                 activations: List[Activation],
                 dropout: List[float],
                 normalize: bool = True):
        super(MultilayerAttention, self).__init__(normalize)

        self.dropout = dropout
        self.activations = activations
        self.hidden_dims = hidden_dims
        self.vector_dim = vector_dim
        self.matrix_dim = matrix_dim

        self.ll1 = Linear(matrix_dim, hidden_dims[0])
        self.ll2 = Linear(vector_dim, hidden_dims[0])

        self.ff = FeedForward(
            input_dim=hidden_dims[0],
            num_layers=len(hidden_dims),
            hidden_dims=hidden_dims[1:] + [1],
            activations=activations[1:] + [Activation.by_name("linear")()],
            dropout=self.dropout[1:] + [0.0]
        )

    def _forward_internal(self, vector: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        """
        Inputs:

        - vector: shape ``(batch_size, embedding_dim)``
        - matrix: shape ``(batch_size, num_rows, embedding_dim)``

        :return: [batch_size, num_rows] - scores for each row
        """

        # Shape: [batch_size, num_rows, first_hidden_dim]
        first_layer_out = self.ll1(matrix) + self.ll2(vector.unsqueeze(1))

        # Shape: [batch_size, num_rows, first_hidden_dim]
        first_activation = self.activations[0](first_layer_out)

        # Shape: [batch_size, num_rows]
        return self.ff(first_activation).squeeze(2)


class LinearDotAttention(Attention):
    def __init__(self, vector_dim: int, matrix_dim: int, normalize: bool = True):
        super(LinearDotAttention, self).__init__(normalize)

        self.ll = Linear(vector_dim, matrix_dim)

    def _forward_internal(self, vector: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        transformed_vectors = self.ll(vector)

        return matrix.bmm(transformed_vectors.unsqueeze(-1)).squeeze(-1)
