from typing import Dict, Tuple, Optional

import torch
from allennlp.data import Vocabulary
from allennlp.modules import TextFieldEmbedder, Embedding


class CandidatesSelector:

    def get_candidates(
            self,
            word: Dict[str, torch.LongTensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Generates synonym candidates for a given words
        In simplest implementation - all words

        :param word:
        :return:
            candidate ids: [batch_size, num_variants]
            candidate vectors: [batch_size, num_variants, emb_size],
            optional mask: [batch_size, num_variants]
        """
        raise NotImplementedError()


class AllVocabCandidates(CandidatesSelector):

    @classmethod
    def _build_mapping(cls, vocab, src_namespace, dist_namespace) -> torch.Tensor:
        """
        Build mapping tensor from ``src_namespace`` indexes to ``dist_namespace``

        :return:
            - [src_vocab_size] - mapping src to dist
        """
        target_to_tokens = dict(
            (index, vocab.get_token_index(token=token, namespace=dist_namespace)) for token, index in
            vocab.get_token_to_index_vocabulary(src_namespace).items()
        )

        d_array = torch.arange(max(vocab.get_token_to_index_vocabulary(src_namespace).values()) + 1)

        target_indexes = torch.tensor(list(target_to_tokens.keys()))
        tokens_indexes = torch.tensor(list(target_to_tokens.values()))

        d_array[target_indexes] = tokens_indexes

        return d_array.long()

    def __init__(self, vocab: Vocabulary, w2v: Embedding):
        self.w2v = w2v
        self.vocab = vocab

        self.target_size = self.vocab.get_vocab_size("target")

        # shape: [target_vocab_size]
        target_to_tokens = self._build_mapping(vocab, 'target', 'tokens').to(device=self.w2v.weight.device)

        self.target_indexes = torch.arange(self.target_size, dtype=torch.long, device=self.w2v.weight.device)

        # shape: [1, vocab_size, embedding_size]
        self.target_vectors = self.w2v.weight[target_to_tokens].unsqueeze(0)

    def get_candidates(
            self,
            word: Dict[str, torch.LongTensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        # shape: [batch_size]
        original_target_ids = word['target']

        batch_size = original_target_ids.size(0)

        # shape: [batch_size, vocab_size]
        mask = original_target_ids.new_ones((original_target_ids.size(0), self.target_size))

        # shape: [batch_size, vocab_size]
        mask = mask.scatter_(1, original_target_ids, 0)

        # shape: [batch_size, vocab_size, embeddings_size]
        batch_candidate_vectors: torch.Tensor = self.target_vectors.expand((batch_size, -1, -1))

        # shape: [batch_size, vocab_size]
        batch_candidate_idxs = self.target_indexes.unsqueeze(0).expand((batch_size, -1))

        return (
            batch_candidate_idxs,
            batch_candidate_vectors,
            mask
        )
