import json
from collections import defaultdict

from word_gan.gan.candidate_selectors.group_selector import ColumnGroupSelector
from word_gan.settings import SETTINGS


class JsonGroupSelector(ColumnGroupSelector):

    def load_mapping(self, file, device):
        with open(file) as fd:
            data = json.load(fd)

        padding_token = self.vocab._padding_token
        padding_idx = self.vocab.get_token_index(padding_token, namespace='target')
        variant_size = SETTINGS.NUM_VARIANTS

        default_idxs, default_mask = self.get_empty_variants(variant_size, device, padding_idx)

        groups = defaultdict(lambda: (default_idxs, default_mask))

        for word, variants in data.items():
            word_idx = self.vocab.get_token_index(word, namespace='target')

            target_tensor, mask_tensor = self.words_to_tensor(variants, device=device, padding_token=padding_token)

            groups[word_idx] = target_tensor, mask_tensor

        return groups
