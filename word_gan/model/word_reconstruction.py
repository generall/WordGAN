from typing import Dict

import torch
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder
from allennlp.training.metrics import Metric, CategoricalAccuracy
from torch import nn

from word_gan.model.embedding_to_word import EmbeddingToWord


class WordReconstruction(Model):

    def __init__(self, w2v: TextFieldEmbedder, v2w: EmbeddingToWord, vocab: Vocabulary):
        super(WordReconstruction, self).__init__(vocab)
        self.v2w = v2w
        self.w2v = w2v

        self.loss = torch.nn.NLLLoss()

        self.accuracy = CategoricalAccuracy()

        self.metrics: Dict[str, Metric] = {
            'accuracy': self.accuracy
        }

    def forward(self, word):

        # size: [batch_size x embedding_dim]
        word_embeddings = self.w2v(word).squeeze(1)

        word_probs = self.v2w(word_embeddings)

        result = {
            'words': word_probs
        }

        if self.training:
            target = word['tokens'].view(-1)
            self.accuracy(word_probs, target)
            result['loss'] = self.loss(word_probs, target)

        return result

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return dict((key, metric.get_metric(reset)) for key, metric in self.metrics.items())

