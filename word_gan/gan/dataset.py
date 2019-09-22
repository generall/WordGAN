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

    def __init__(self, dict_path, limit_words=-1, limit_freq=0, max_context_size: int = 2,
                 token_indexers: Dict[str, TokenIndexer] = None):
        """

        :param dict_path: path to the dict of acceptable fords to change
        :param limit_words: Max word count from dictionary
        :param limit_freq: Minimum frequency of words
        :param max_context_size:
        """
        super().__init__(lazy=True)
        self.max_context_size = max_context_size
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

        left_context, right_context = self.get_context(tokens, idx, self.max_context_size)

        if len(left_context) < self.max_context_size:
            left_context = [self.left_padding] + left_context
        if len(right_context) < self.max_context_size:
            right_context = right_context + [self.right_padding]

        left_context = TextField(left_context, self.token_indexers)
        right_context = TextField(right_context, self.token_indexers)

        target_token_field = TextField([target_word], self.target_indexer)

        return Instance({
            "left_context": left_context,
            "right_context": right_context,
            "word": target_token_field
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