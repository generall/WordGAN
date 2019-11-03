import os
from unittest import TestCase

import torch
from allennlp.data import Vocabulary
from loguru import logger

from word_gan.gan.candidate_selectors.group_selector import GroupSelector
from word_gan.gan.helpers.loaders import load_w2v
from word_gan.settings import VerbsSettings


class TestGroupSelector(TestCase):
    def test_get_candidates(self):
        vocab = Vocabulary.from_files(VerbsSettings.VOCAB_PATH)

        self.assertIsNotNone(vocab.get_token_index('imbibing', namespace='target'))

        logger.info("Vocab loaded")

        w2v = load_w2v(
            weights_file=os.path.join(VerbsSettings.DATA_DIR, 'target_vectors.txt'),
            vocab=vocab,
            namespace='target'
        )

        logger.info("w2v loaded")

        selector = GroupSelector(
            vocab=vocab,
            target_w2v=w2v,
            groups_file=os.path.join(VerbsSettings.DATA_DIR, 'verbs.txt')
        )

        logger.info("selector loaded")

        candidates_idx, candidate_vectors, candidates_mask = selector.get_candidates(word={
            "target": torch.LongTensor([
                vocab.get_token_index('ran', namespace='target'),
                vocab.get_token_index('eat', namespace='target'),
                vocab.get_token_index('dry', namespace='target')
            ])
        })

        self.assertEqual(len(candidates_mask.shape), 2)

        self.assertLess(candidates_mask.sum(), candidates_mask.size(0) * candidates_mask.size(1))

        self.assertEqual(candidates_idx.shape, candidates_mask.shape)

        self.assertEqual(candidates_idx.size(0), 3)

        self.assertEqual(len(candidate_vectors.shape), 3)


