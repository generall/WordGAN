from collections import defaultdict
from typing import Dict, Tuple, Optional, List

import pandas as pd
import torch
from allennlp.data import Vocabulary
from allennlp.modules import Embedding

from word_gan.gan.candidate_selectors.base_selector import CandidatesSelector


class GroupSelector(CandidatesSelector):
    """
    Here we save a multiple groups of candidates, if some word is present is some group
    all other group members are candidates for prediction
    """

    def __init__(self, vocab: Vocabulary, target_w2v: Embedding, groups_file: str, device=None):
        self.target_w2v = target_w2v
        self.vocab = vocab

        self.groups = self.load_columns(groups_file, device or self.target_w2v.weight.device)

        # Mapping target_id -> candidate_ids
        # Shape: [target_size, groups_size]
        self.mapping_ids = torch.stack([
            self.groups[target_id][0] for target_id in range(self.vocab.get_vocab_size('target'))
        ])

        # Shape: [target_size, groups_size]
        self.mapping_mask = torch.stack([
            self.groups[target_id][1] for target_id in range(self.vocab.get_vocab_size('target'))
        ])

        self.target_size = self.vocab.get_vocab_size("target")

    @classmethod
    def get_empty_variants(cls, variants_size, device, padding_idx):
        mask = torch.zeros((variants_size,), dtype=torch.long, device=device)
        idxs = torch.full((variants_size,),
                          fill_value=padding_idx,
                          dtype=torch.long, device=device)

        return idxs, mask

    def words_to_tensor(self, words: List[str], device, padding_token, pad_length=None) -> (torch.Tensor, torch.Tensor):

        pad_length = pad_length or len(words)
        delta_length = pad_length - len(words)

        words = words + [padding_token] * delta_length

        target_indexes = [self.vocab.get_token_index(token=token, namespace='target') for token in words]
        target_tensor = torch.tensor(target_indexes, dtype=torch.long, device=device)

        mask = (target_tensor != self.vocab.get_token_index(padding_token, namespace='target')).long()

        return target_tensor, mask

    def load_columns(self, file, device):
        df = pd.read_csv(file, sep='\t')

        padding_token = self.vocab._padding_token
        padding_idx = self.vocab.get_token_index(padding_token, namespace='target')
        variant_size = df.shape[0]

        default_idxs, default_mask = self.get_empty_variants(variant_size, device, padding_idx)

        groups = defaultdict(lambda: (default_idxs, default_mask))

        for column in df.columns:
            group = list(df[column])

            token_tensor, target_tensor = self.words_to_tensor(group, device=device, padding_token=padding_token)

            for word in df[column]:
                word_idx = self.vocab.get_token_index(word, namespace='target')
                if word_idx not in groups:
                    groups[word_idx] = token_tensor, target_tensor

        return groups

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

        # shape: [batch_size, 1]
        original_target_ids = word['target'].squeeze()

        # shape: [batch_size, num_variants]
        candidate_ids = self.mapping_ids[original_target_ids]

        # shape: [batch_size, num_variants]
        mask = self.mapping_mask[original_target_ids]

        # shape: [batch_size, num_variants]
        mask = mask & (candidate_ids != original_target_ids.unsqueeze(-1)).long()

        # shape: [batch_size, num_variants, emb_size]
        candidate_vectors = self.target_w2v(candidate_ids)

        return (
            candidate_ids,
            candidate_vectors,
            mask
        )
