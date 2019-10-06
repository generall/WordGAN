from typing import List
import random

import torch
from allennlp.data import Vocabulary


class TrainLogger:

    def __init__(self, serialization_path: str, batch_period: int):
        self.batch_period = batch_period
        self.serialization_path = serialization_path

    def _generate_info(self, batch, result) -> List[str]:
        raise NotImplementedError()

    def log_info(self, batch, result, batch_num):
        if batch_num % self.batch_period == 0:
            with open(self.serialization_path, 'a') as out:
                for info in self._generate_info(batch, result):
                    out.write(f"{batch_num}. {info}\n")


class WordGanLogger(TrainLogger):

    def __init__(self, serialization_path: str, batch_period: int, vocab: Vocabulary):
        super().__init__(serialization_path, batch_period)
        self.vocab = vocab

    def _to_words(self, tsr: torch.Tensor):
        idxs = tsr.detach().cpu().tolist()
        words = [self.vocab.get_token_from_index(x) for x in idxs]
        return ' '.join(words)

    def _generate_info(self, batch, result) -> List[str]:
        batch_size = batch['word']['tokens'].size(0)
        sample_idx = random.randint(0, batch_size)

        left_context = self._to_words(batch['left_context']['tokens'][sample_idx])
        right_context = self._to_words(batch['right_context']['tokens'][sample_idx])
        word = self._to_words(batch['word']['tokens'][sample_idx])

        synonym = self._to_words(result['generated_indexes']['token_indexes'][[sample_idx]])

        return [" ".join([left_context, word, right_context, '->', synonym])]
