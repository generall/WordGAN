import os

from loguru import logger

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))


class BaseSettings:
    # tokens: All tokens, present in Word2Vec.
    # target: Changeable words
    DATA_DIR = os.path.join(ROOT_DIR, 'data')
    VOCAB_PATH = os.path.join(DATA_DIR, 'vocab')
    EMBEDDINGS_SIZE = 300
    BATCH_SIZE = 64


class TestSettings(BaseSettings):
    EMBEDDINGS_SIZE = 300
    DATA_DIR = os.path.join(ROOT_DIR, 'data', 'tests')
    VOCAB_PATH = os.path.join(DATA_DIR, 'vocab')


class SyntheticSettings(BaseSettings):  # used for experiments on a synthetic dataset
    EMBEDDINGS_SIZE = 20
    DATA_DIR = os.path.join(ROOT_DIR, 'data', 'synthetic')
    VOCAB_PATH = os.path.join(DATA_DIR, 'vocab')


class VerbsSettings(BaseSettings):
    EMBEDDINGS_SIZE = 300
    DATA_DIR = os.path.join(ROOT_DIR, 'data', 'verbs')
    VOCAB_PATH = os.path.join(DATA_DIR, 'vocab')


mode = os.getenv('MODE', 'verbs')

if mode == 'test':
    logger.info("USING TEST SETTINGS")
    settings_class = TestSettings
elif mode == 'verbs':
    logger.info("USING VERBS SETTINGS")
    settings_class = VerbsSettings
elif mode == 'synthetic':
    logger.info("USING SYNTHETIC SETTINGS")
    settings_class = SyntheticSettings
else:
    settings_class = BaseSettings

SETTINGS = settings_class
