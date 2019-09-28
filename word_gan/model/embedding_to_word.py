from torch import nn
from torch.nn import Parameter
import torch


class EmbeddingToWord(nn.Module):
    """
    This module is used to restore actual word from embedding.
    """

    def __init__(self, embedding_size, words_count):
        super(EmbeddingToWord, self).__init__()

        self.embedding_size = embedding_size
        self.words_count = words_count

        self.norm_weights = Parameter(torch.FloatTensor(self.words_count, self.embedding_size))

        self.norm_weights.requires_grad = False

    @classmethod
    def _norm_tensor(cls, vectors: torch.FloatTensor):
        """
        Normalize each vector in batch. Length of all vectors will be 1

        :param vectors: [batch_size, vector_length]
        :return:
        """
        return vectors / vectors.norm(p=2, dim=1, keepdim=True)

    def init_from_embeddings(self, embeddings: torch.FloatTensor):
        """

        :param embeddings: word2vec embeddings in shape: [word_count, embedding_size]
        :return:
        """
        self.norm_weights.copy_(self._norm_tensor(embeddings))

    def forward(self, vectors: torch.FloatTensor):
        """

        :param vectors: [batch_size x embedding_size]
        :return: [batch_size x word_count] in range (-1, 1)
        """

        # shape: [vector_size, batch_size]
        norm_vectors = self._norm_tensor(vectors).transpose(1, 0)

        # shape: [batch_size x word_count]
        return torch.mm(self.norm_weights, norm_vectors).transpose(0, 1)


