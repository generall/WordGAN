import os
import sys

from allennlp.data import Vocabulary

from word_gan.gan.dataset import TextDatasetReader
from word_gan.settings import DATA_DIR, SETTINGS


def load_vocab(target_vocab_path, tokens_vocab_path):
    vocab = Vocabulary()

    vocab.set_from_file(
        filename=target_vocab_path,
        is_padded=False,
        namespace='target'
    )

    vocab.set_from_file(
        filename=tokens_vocab_path,
        is_padded=True,
        namespace='tokens'
    )

    return vocab


if __name__ == '__main__':

    freq_dict_path = os.path.join(DATA_DIR, 'count_1w.txt')

    if len(sys.argv) > 1:
        text_data_path = sys.argv[1]
    else:
        text_data_path = os.path.join(DATA_DIR, 'test_data.txt')

    vocab = load_vocab(SETTINGS.TARGET_VOCAB_PATH, SETTINGS.TOKEN_VOCAB_PATH)

    print('target size', vocab.get_vocab_size('target'))
    print('tokens size', vocab.get_vocab_size('tokens'))

    reader = TextDatasetReader(
        dict_path=freq_dict_path,
        limit_words=100_000,
        limit_freq=0,
        small_context=1,
        large_context=2
    )

    train_dataset = reader.read(text_data_path)




