import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'data')

TEST_DATA_DIR = os.path.join(DATA_DIR, 'tests')


class BaseSettings:
    TOKEN_VOCAB_PATH = os.path.join(DATA_DIR, 'tokens_vocab.txt')  # All tokens, present in Word2Vec
    TARGET_VOCAB_PATH = os.path.join(DATA_DIR, 'target_vocab.txt')  # Changeable words


SETTINGS = BaseSettings
