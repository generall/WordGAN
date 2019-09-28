from typing import Dict, Iterable

from allennlp.data import DatasetReader, TokenIndexer, Instance, Token
from allennlp.data.fields import TextField
from allennlp.data.token_indexers import SingleIdTokenIndexer


class DictDatasetReader(DatasetReader):
    """
    Reads simple frequency lists in format <word> <freq>
    """

    def __init__(self, limit_words=-1, limit_freq=0, token_indexers: Dict[str, TokenIndexer] = None,
                 namespace='tokens') -> None:
        super().__init__(lazy=False)
        self.limit_freq = limit_freq
        self.limit_words = limit_words
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer(namespace=namespace)}

    def text_to_instance(self, token: str) -> Instance:
        word_field = TextField([Token(token)], self.token_indexers)
        fields = {"word": word_field, "target": word_field}
        return Instance(fields)

    def _read(self, file_path: str) -> Iterable[Instance]:
        with open(file_path) as fd:
            for idx, line in enumerate(fd):
                word, freq = line.strip().split()
                freq = int(freq)
                if idx == self.limit_words:
                    break
                if freq < self.limit_freq:
                    break
                yield self.text_to_instance(word)
