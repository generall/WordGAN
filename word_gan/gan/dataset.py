from typing import Iterable, Dict

from allennlp.data import DatasetReader, Instance, TokenIndexer
from allennlp.data.fields import TextField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from nltk.tokenize import WordPunctTokenizer


class TextDatasetReader(DatasetReader):
    """
    Reads raw text, finds replaceable words, generates instance: word with context
    """

    @classmethod
    def read_dict(cls, file_path, limit_words=-1, limit_freq=0):
        word_dict = {}
        with open(file_path) as fd:
            for idx, line in enumerate(fd):
                word, freq = line.strip().split()
                freq = int(freq)
                if idx == limit_words:
                    break
                if freq < limit_freq:
                    break

                word_dict[word] = freq

        return word_dict

    def __init__(self, dict_path, limit_words=-1, limit_freq=0, small_context=1, large_context=2,
                 token_indexers: Dict[str, TokenIndexer] = None):
        """

        :param dict_path: path to the dict of acceptable fords to change
        :param limit_words: Max word count from dictionary
        :param limit_freq: Minimum frequency of words
        :param small_context:
        :param large_context:
        """
        super().__init__(lazy=True)
        self.large_context_size = large_context
        self.small_context_size = small_context
        self.word_dict = self.read_dict(dict_path, limit_words, limit_freq)

        self.tokenizer = WordPunctTokenizer()
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

        self.target_indexer = {"target": SingleIdTokenIndexer(
            namespace='target',
            lowercase_tokens=True
        )}

        self.left_padding = 'BOS'
        self.right_padding = 'EOS'

    def text_to_instance(self, tokens, idx) -> Instance:
        target_word = tokens[idx]

        small_left_context, small_right_context = self.get_context(tokens, idx, self.small_context_size)
        large_left_context, large_right_context = self.get_context(tokens, idx, self.large_context_size)

        if len(small_left_context) < self.small_context_size:
            small_left_context = [self.left_padding] + small_left_context
        if len(small_right_context) < self.small_context_size:
            small_right_context = small_right_context + [self.right_padding]

        if len(large_left_context) < self.large_context_size:
            large_left_context = [self.left_padding] + large_left_context
        if len(large_right_context) < self.large_context_size:
            large_right_context = large_right_context + [self.right_padding]

        small_context = small_left_context + [target_word] + small_right_context

        small_context_field = TextField(small_context, self.token_indexers)

        large_context = large_left_context + [target_word] + large_right_context

        large_context_field = TextField(large_context, self.token_indexers)

        target_token_field = TextField([target_word], self.target_indexer)

        return Instance({
            "small_context": small_context_field,
            "large_context": large_context_field,
            "target_word": target_token_field
        })

    @classmethod
    def get_context(cls, tokens, idx, size):
        """
        >>> TextDatasetReader.get_context([1,2,3,4,5,7], 1, 2)
        ([1], [3, 4])

        >>> TextDatasetReader.get_context([1,2,3,4,5,7], 4, 2)
        ([3, 4], [7])


        :param tokens:
        :param idx:
        :param size:
        :return:
        """
        return tokens[max(idx - size, 0):idx], tokens[idx + 1:idx + size + 1]

    def _read(self, file_path: str) -> Iterable[Instance]:
        with open(file_path) as fd:
            for line in fd:
                tokens = self.tokenizer.tokenize(line)
                for idx, token in enumerate(tokens):
                    if token in self.word_dict:
                        yield self.text_to_instance(tokens, idx)
