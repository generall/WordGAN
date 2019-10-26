from unittest import TestCase
import torch
from allennlp.data import Vocabulary

from word_gan.gan.generator import Generator


class TestGenerator(TestCase):
    def test__get_loss_mask(self):

        torch.set_printoptions(precision=2)

        target_indexes = torch.tensor([2, 5, 6, 1, 4])

        synonym_words_score = torch.tensor([
            [0.24, 0.75, 0.99, 0.12, 0.42, 0.25, 0.53, 0.87, 0.14, 0.88],
            [0.89, 0.31, 0.10, 0.88, 0.07, 0.70, 0.16, 0.28, 0.81, 0.68],
            [0.79, 0.22, 0.70, 0.39, 0.19, 0.05, 0.70, 0.20, 0.81, 0.38],
            [0.08, 0.85, 0.92, 0.57, 0.76, 0.53, 0.62, 0.06, 0.94, 0.87],
            [0.60, 0.55, 0.65, 0.23, 0.33, 0.07, 0.04, 0.21, 0.19, 0.54]
        ])

        print("")

        print(synonym_words_score)

        print("")

        loss_mask, max_scores = Generator._get_loss_mask(
            target_indexes=target_indexes,
            synonym_words_score=synonym_words_score,
            synonym_delta=0.1
        )

        self.assertEqual(loss_mask.shape[0], 5)
        self.assertEqual(max_scores.shape[0], 5)

        diff = loss_mask.long() - torch.tensor([1, 0, 0, 1, 0])

        diff = diff.sum()

        self.assertEqual(diff, 0)

        print(loss_mask)
        print(max_scores)

    def test_build_vocab_mapping(self):

        vocab = Vocabulary({
            'target': {
                'aaa': 1,
                'bbb': 1,
                'ccc': 1,
                'ddd': 1,
                'eee': 1,
            },
            'tokens': {
                '111': 1,
                'aaa': 1,
                '222': 1,
                'bbb': 1,
                'ccc': 1,
                '333': 1,
                'ddd': 1,
                'eee': 1,
            }
        })

        mapping = Generator._build_mapping(vocab, 'target', 'tokens')

        print(mapping)

        self.assertEqual(mapping[vocab.get_token_index('ccc', 'target')], vocab.get_token_index('ccc', 'tokens'))
        self.assertNotEqual(vocab.get_token_index('ccc', 'target'), vocab.get_token_index('ccc', 'tokens'))
