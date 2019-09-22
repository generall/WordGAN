import os
import sys

from allennlp.data import Vocabulary

from word_gan.settings import DATA_DIR
from word_gan.word_reconstruction.dataset import DictDatasetReader

if __name__ == '__main__':

    if len(sys.argv) < 2:
        data_path = os.path.join(DATA_DIR, 'count_1w.txt')
    else:
        data_path = sys.argv[1]

    num_words = int(os.getenv("NUM_WORDS", 100_000))

    reader = DictDatasetReader(limit_words=num_words)
    train_dataset = reader.read(data_path)

    vocab = Vocabulary.from_instances(train_dataset)

    vocab.save_to_files(os.path.join(DATA_DIR, 'v2w_vocab'))

    print('tokens size:', vocab.get_vocab_size('tokens'))
    print('target size:', vocab.get_vocab_size('target'))
