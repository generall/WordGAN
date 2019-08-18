from typing import Dict

import torch
from allennlp.data import Vocabulary
from allennlp.models import Model

from word_gan.model.discriminator import Discriminator
from word_gan.model.embedding_to_word import EmbeddingToWord
from word_gan.model.synonyms_generator import SynonymGenerator
from word_gan.model.word_reconstruction import WordReconstruction


class Generator(Model):

    def __init__(
            self,
            embedding_dim,
            v2w: EmbeddingToWord,
            vocab: Vocabulary
    ):
        """

        :param embedding_dim:
        :param v2w: vector to word mapping layer. Not trainable
        :param vocab:
        """
        super().__init__(vocab)

        self.v2w = v2w
        self.synonyms_generator = SynonymGenerator(embedding_dim)

        self.generator_context_size = 1
        self.discriminator_context_size = 2

    def forward(
            self, left_context, word, right_context, discriminator: Discriminator = None) -> Dict[str, torch.Tensor]:
        """

        :param left_context: [batch_size, context_size, embedding_size]
        :param word: vector of the word of interest [batch_size, embedding_size]
        :param right_context: [batch_size, context_size, embedding_size]
        :param discriminator:
        :return:
        """
        generator_left_context = left_context[:, -self.generator_context_size:, :]
        generator_right_context = right_context[:, :self.generator_context_size, :]
        
        # shape: [batch_size, embedding_size]
        synonym_vectors = self.synonyms_generator(generator_left_context, word, generator_right_context)

        # shape: [batch_size, vocab_size]
        synonym_words_score = self.v2w(synonym_vectors)

        if discriminator:
            discriminator_left_context = left_context[:, -self.discriminator_context_size:, :]
            discriminator_right_context = right_context[:, :self.discriminator_context_size, :]

            # ToDo
            