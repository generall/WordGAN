from unittest import TestCase

import torch

from word_gan.model.discriminator import Discriminator
from word_gan.model.synonyms_generator import SynonymGenerator


class TestSynonymGenerator(TestCase):
    def test_forward(self):
        generator = SynonymGenerator(10)

        batch_size = 5
        vector_size = 10

        t1 = torch.rand(batch_size, 2, vector_size)
        t2 = torch.rand(batch_size, 3, vector_size)

        word = torch.rand(batch_size, vector_size)

        res = generator.forward(
            left_context=t1,
            word=word,
            right_context=t2
        )

        self.assertEqual(res.shape[0], batch_size)
        self.assertEqual(res.shape[1], vector_size)
