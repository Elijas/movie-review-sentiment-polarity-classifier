from pathlib import Path

import joblib
from sklearn.model_selection import train_test_split

from src.preparation.constants import POS, NEG

# Paths for files
ROOT_FOLDERPATH = Path.cwd().parent.parent
RAW_CORPUS_FOLDERPATH = ROOT_FOLDERPATH / 'data' / 'raw'
NEG_RAW_CORPUS_FILEPATH = RAW_CORPUS_FOLDERPATH / 'rt-polarity.neg'
POS_RAW_CORPUS_FILEPATH = RAW_CORPUS_FOLDERPATH / 'rt-polarity.pos'
SPLIT_RAW_DATASET_FILEPATH = ROOT_FOLDERPATH / 'data' / 'processed' / 'split-raw-dataset.joblib'

# Magic values
RANDOMNESS_SEED = 834
DATASET_TEST_SPLIT_RATIO = 0.2

import collections

Dataset = collections.namedtuple('Dataset', 'train test')
DatasetVariables = collections.namedtuple('DatasetVariables', 'x y')


def load_raw_corpus_posneg():
    with NEG_RAW_CORPUS_FILEPATH.open() as file:
        corpus_neg = file.readlines()

    with POS_RAW_CORPUS_FILEPATH.open() as file:
        corpus_pos = file.readlines()

    return corpus_pos, corpus_neg


def load_dataset():
    if SPLIT_RAW_DATASET_FILEPATH.exists():
        with SPLIT_RAW_DATASET_FILEPATH.open('rb') as file:
            dataset = joblib.load(file)
            return dataset

    corpus_pos, corpus_neg = load_raw_corpus_posneg()
    corpus = corpus_neg + corpus_pos
    labels = [NEG] * len(corpus_neg) + [POS] * len(corpus_pos)
    x_trn, x_tst, y_trn, y_tst = train_test_split(corpus, labels,
                                                  test_size=DATASET_TEST_SPLIT_RATIO,
                                                  random_state=RANDOMNESS_SEED)
    dataset = Dataset(train=DatasetVariables(x=x_trn, y=y_trn), test=DatasetVariables(x=x_tst, y=y_tst))

    with SPLIT_RAW_DATASET_FILEPATH.open('wb') as file:
        joblib.dump(dataset, file)

    return dataset
