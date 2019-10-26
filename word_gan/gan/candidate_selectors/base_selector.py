from typing import Dict, Tuple, Optional

import torch


class CandidatesSelector:

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
