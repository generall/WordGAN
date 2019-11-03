from unittest import TestCase
import torch
from allennlp.data import Vocabulary

from word_gan.gan.candidate_selectors.base_selector import CandidatesSelector
from word_gan.gan.generator import Generator


class TestGenerator(TestCase):

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

        mapping = CandidatesSelector._build_mapping(vocab, 'target', 'tokens')

        print(mapping)

        self.assertEqual(mapping[vocab.get_token_index('ccc', 'target')], vocab.get_token_index('ccc', 'tokens'))
        self.assertNotEqual(vocab.get_token_index('ccc', 'target'), vocab.get_token_index('ccc', 'tokens'))
