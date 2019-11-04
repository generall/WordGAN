import os

import pandas as pd

from word_gan.settings import SETTINGS

ORIG_VERB_FORMS_FILE = os.path.join(SETTINGS.DATA_DIR, 'groups.txt')

TARGET_PATH = os.path.join(SETTINGS.VOCAB_PATH, 'target.txt')


def convert_verbs(path, out_path):
    with open(out_path, 'w') as out:
        # out.write('@@PADDING@@\n')
        out.write('@@UNKNOWN@@\n')

        all_words = set()

        with open(path) as fd:
            for line in fd:
                for word in line.strip().split():
                    all_words.add(word)

        for word in all_words:
            out.write(word)
            out.write('\n')


if __name__ == '__main__':
    convert_verbs(ORIG_VERB_FORMS_FILE, TARGET_PATH)
