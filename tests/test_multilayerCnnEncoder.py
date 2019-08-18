from unittest import TestCase

import torch

from word_gan.model.multilayer_cnn import MultilayerCnnEncoder


class TestMultilayerCnnEncoder(TestCase):
    def test_forward(self):
        tensor = torch.rand(4, 5, 10)

        filters_size = 7
        ngram_filter_sizes = (2, 3)

        cnn_layer = MultilayerCnnEncoder(
            embedding_dim=10,
            num_filters=filters_size,
            layers=2,
            ngram_filter_sizes=ngram_filter_sizes,
            # output_dim=7
        )

        res = cnn_layer(tensor)

        print(res.shape)

        self.assertEqual(res.shape[0], 4)
        self.assertEqual(res.shape[1], filters_size * len(ngram_filter_sizes))
