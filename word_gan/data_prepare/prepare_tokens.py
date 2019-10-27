import os

from word_gan.settings import SETTINGS

MODEL_PATH = os.path.join(SETTINGS.DATA_DIR, 'model.txt')

TOKENS_PATH = os.path.join(SETTINGS.VOCAB_PATH, 'tokens.txt')


def extract_tokens_from_model(path, out_path):
    with open(out_path, 'w') as out:
        # out.write('@@PADDING@@\n')
        out.write('@@UNKNOWN@@\n')
        out.write('EOS\n')
        out.write('BOS\n')

        with open(path) as fd:
            next(fd)
            for line in fd:
                word, _ = line.split(' ', 1)
                out.write(word)
                out.write('\n')


if __name__ == '__main__':
    extract_tokens_from_model(MODEL_PATH, TOKENS_PATH)
