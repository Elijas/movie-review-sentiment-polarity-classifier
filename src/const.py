"""
Constants and settings
"""
import os
from pathlib import Path

RANDOMNESS_SEED = 834
DATASET_TEST_SPLIT_RATIO = 0.20


class PATHS:
    ROOT_FOLDER = Path(os.path.dirname(os.path.realpath(__file__))).parent
    MODEL_FOLDER = ROOT_FOLDER / 'model'
    DATA_FOLDER = ROOT_FOLDER / 'data'
    CACHE_FOLDER = ROOT_FOLDER / 'cache'

    RAW_DATASET_FOLDER = DATA_FOLDER / 'raw'
    CORPUS_POS = RAW_DATASET_FOLDER / 'rt-polarity.pos'
    CORPUS_NEG = RAW_DATASET_FOLDER / 'rt-polarity.neg'

    RAW_STRUCTURED_DATASET_FOLDER = DATA_FOLDER / 'raw_structured'
    RAW_STRUCTURED_DATASET_TRN = RAW_STRUCTURED_DATASET_FOLDER / 'dataset-train.joblib'
    RAW_STRUCTURED_DATASET_TST = RAW_STRUCTURED_DATASET_FOLDER / 'dataset-test.joblib'


class JUPYTER:
    FIGURE_SIZE = (12, 6)


class LABELS:
    NEG = 0
    POS = 1
    TITLES = {
        NEG: "Negative",
        POS: "Positive"
    }
