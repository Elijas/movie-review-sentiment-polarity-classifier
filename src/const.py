import os
from pathlib import Path

RANDOMNESS_SEED = 834
DATASET_TEST_SPLIT_RATIO = 0.20


class PATHS:
    ROOT_FOLDER = Path(os.path.dirname(os.path.realpath(__file__))).parent
    CORPUS_FOLDER = ROOT_FOLDER / 'data'
    CORPUS_POS = CORPUS_FOLDER / 'rt-polarity.pos'
    CORPUS_NEG = CORPUS_FOLDER / 'rt-polarity.neg'
    SPLIT_DATASET_FOLDER = CORPUS_FOLDER / 'split'
    SPLIT_DATASET_TRN = SPLIT_DATASET_FOLDER / 'dataset-train.joblib'
    SPLIT_DATASET_TST = SPLIT_DATASET_FOLDER / 'dataset-test.joblib'


class JUPYTER:
    FIGURE_SIZE = (12, 6)


class LABELS:
    NEG = 0
    POS = 1
    TITLES = {
        NEG: "Negative",
        POS: "Positive"
    }
