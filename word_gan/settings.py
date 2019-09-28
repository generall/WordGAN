import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'data')

TEST_DATA_DIR = os.path.join(DATA_DIR, 'tests')


class BaseSettings:
    # tokens: All tokens, present in Word2Vec.
    # target: Changeable words
    VOCAB_PATH = os.path.join(DATA_DIR, 'vocab')


SETTINGS = BaseSettings
