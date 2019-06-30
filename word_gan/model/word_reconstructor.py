from torch import nn
import torch


class WordReconstructor(nn.Module):
    """
    This module is used to restore actual word from embedding.
    """

    def __init__(self, embedding_size, words_count):
        super(WordReconstructor, self).__init__()

        self.embedding_size = embedding_size
        self.words_count = words_count

        self.weights = nn.Linear(embedding_size, words_count)
        self.softmax = nn.LogSoftmax()

    def forward(self, embeddings):
        """

        >>> list(WordReconstructor(2, 10).forward(torch.tensor([[[1,2], [3,4], [5,6]]]).float()).shape)
        [1, 3, 10]

        :param embeddings: [.. x ( .... ) .. x embedding_size]
        :return: [.. x ( .... ) .. x words_count]
        """
        return self.softmax(self.weights(embeddings))
