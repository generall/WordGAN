import itertools
import os
from typing import Iterable, Dict

from allennlp.data import DatasetReader, Instance, TokenIndexer, Token, Vocabulary
from allennlp.data.fields import TextField
from allennlp.data.iterators import BasicIterator
from allennlp.data.token_indexers import SingleIdTokenIndexer
from nltk.tokenize import WordPunctTokenizer

from word_gan.gan.fasttext_indexer import StaticFasttextTokenIndexer
from word_gan.settings import SETTINGS


class TextDatasetReader(DatasetReader):
    """
    Reads raw text, finds replaceable words, generates instance: word with context
    """

    @classmethod
    def read_dict(cls, file_path, limit_words=-1, limit_freq=0):
        word_dict = {}
        with open(file_path) as fd:
            for idx, line in enumerate(fd):
                word, *freq = line.strip().split()

                if idx == limit_words:
                    break

                if len(freq) > 0:
                    freq = freq[0]
                    freq = int(freq)
                    if freq < limit_freq:
                        break
                else:
                    freq = 1

                word_dict[word] = freq

        return word_dict

    def __init__(
            self,
            dict_path,
            limit_words=-1,
            limit_freq=0,
            max_context_size: int = 4,
            token_indexers: Dict[str, TokenIndexer] = None,
            target_indexers: Dict[str, TokenIndexer] = None
    ):
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

        self.target_indexer = target_indexers or {
            "target": SingleIdTokenIndexer(
                namespace='target',
                lowercase_tokens=True
            ),
            "tokens": SingleIdTokenIndexer()
        }

        self.left_padding = 'BOS'
        self.right_padding = 'EOS'

    def text_to_instance(self, tokens, idx) -> Instance:

        target_word = tokens[idx]

        left_context, right_context = self.get_context(tokens, idx, self.max_context_size)

        if len(left_context) < self.max_context_size:
            left_context = [self.left_padding] + left_context
        if len(right_context) < self.max_context_size:
            right_context = right_context + [self.right_padding]

        left_context = TextField([Token(token) for token in left_context], self.token_indexers)
        right_context = TextField([Token(token) for token in right_context], self.token_indexers)

        target_token_field = TextField([Token(target_word)], self.target_indexer)

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


if __name__ == '__main__':
    freq_dict_path = os.path.join(SETTINGS.VOCAB_PATH, 'target.txt')

    vocab = Vocabulary.from_files(SETTINGS.VOCAB_PATH)

    print('target size', vocab.get_vocab_size('target'))
    print('tokens size', vocab.get_vocab_size('tokens'))

    fasttext_indexer = StaticFasttextTokenIndexer(
        model_path=os.path.join(SETTINGS.DATA_DIR, 'shrinked_fasttext.model'),
        namespace='tokens',
        lowercase_tokens=True
    )

    reader = TextDatasetReader(
        dict_path=freq_dict_path,
        limit_words=-1,
        limit_freq=0,
        max_context_size=3,
        token_indexers={
            "tokens": fasttext_indexer
        },
        target_indexers={
            "tokens": fasttext_indexer,
            "target": SingleIdTokenIndexer(
                namespace='target',
                lowercase_tokens=True
            )
        },
    )

    text_data_path = os.path.join(SETTINGS.DATA_DIR, 'train_data.txt')

    train_dataset = reader.read(text_data_path)

    iterator = BasicIterator(batch_size=4)

    iterator.index_with(vocab)

    data_iterator = iterator(train_dataset, num_epochs=1)

    for idx, batch in itertools.islice(enumerate(data_iterator), 1):
        print(idx)

        print(batch['left_context']['tokens'])
        print(batch['word']['tokens'])
        print(batch['right_context']['tokens'])
