"""
Constants and settings
"""
import os
from pathlib import Path

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

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
    MODEL_FOLDER = ROOT_FOLDER / 'model'


class JUPYTER:
    FIGURE_SIZE = (12, 6)


class LABELS:
    NEG = 0
    POS = 1
    TITLES = {
        NEG: "Negative",
        POS: "Positive"
    }


class TRAINING_GS_OPTS:
    NAIVE_BAYES = {
        'estimator': Pipeline([
            ('vect', CountVectorizer(ngram_range=(1, 3))),
            ('tfidf', TfidfTransformer()),
            ('clf', MultinomialNB())
        ]),
        'param_grid': {
            'vect__ngram_range': [(1, 3)],
            'tfidf__use_idf': [True],  # , False
            'tfidf__norm': ['l1'],  # , 'l2'
            'clf__alpha': [1e-3],  # , 1e-2, 1e-1, 1
        },
        'cv': 3,
        'scoring': 'accuracy',
        'verbose': 5,
        'n_jobs': -1,
    }
