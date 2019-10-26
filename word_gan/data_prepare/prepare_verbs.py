import os

import pandas as pd

from word_gan.settings import ROOT_DIR, VerbsSettings

ORIG_VERB_FORMS_FILE = os.path.join(ROOT_DIR, 'data', 'verbs', 'verbs.txt')

TARGET_PATH = os.path.join(VerbsSettings.VOCAB_PATH, 'target.txt')


def convert_verbs(path, out_path):
    df = pd.read_csv(path, sep='\t')

    with open(out_path, 'w') as out:
        out.write('@@PADDING@@\n')
        out.write('@@UNKNOWN@@\n')

        for column in df.columns:
            for word in df[column]:
                out.write(word)
                out.write('\n')


if __name__ == '__main__':
    convert_verbs(ORIG_VERB_FORMS_FILE, TARGET_PATH)
