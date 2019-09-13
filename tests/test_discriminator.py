from unittest import TestCase

import torch

from word_gan.model.synonym_discriminator import SynonymDiscriminator


class TestDiscriminator(TestCase):
    def test_forward(self):
        descriminator = SynonymDiscriminator(10, vocab=None)

        batch_size = 5

        t1 = torch.rand(batch_size, 2, 10)
        t2 = torch.rand(batch_size, 3, 10)

        word = torch.rand(batch_size, 10)

        labels = (torch.rand(batch_size) > 0.5).float()

        res = descriminator.forward(
            left_context=t1,
            word=word,
            right_context=t2,
            labels=labels
        )

        self.assertIn('loss', res)

        res = descriminator.forward(
            left_context=t1,
            word=word,
            right_context=t2
        )

        self.assertIn('scores', res)

        self.assertEqual(res['scores'].shape[0], batch_size)
