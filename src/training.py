"""
Functions and constants for model hyperparameter search and training
"""
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from src.processing import alpha__analyzer_fun, alpha_stem__analyzer_fun, \
    alpha_stem_stop__analyzer_fun

_ALL_CORES = -1


def _get_shared_gs_opts(dry_run: bool):
    return {
        'cv': 10,
        'scoring': 'f1' if not dry_run else 'accuracy',
        'verbose': 1,
        'n_jobs': _ALL_CORES,
    }


_SHARED_PIPELINE_ELEMENTS = [
    ('vect', CountVectorizer(lowercase=True, strip_accents='ascii')),
    ('tfidf', TfidfTransformer()),
]


def _get_shared_param_grid_opts(dry_run: bool):
    return {
        'vect__min_df': [None, 2, 5] if not dry_run else [1, 2],
        'vect__analyzer': [
            'word',
            alpha__analyzer_fun,
            alpha_stem__analyzer_fun,
            alpha_stem_stop__analyzer_fun
        ],
        'vect__ngram_range': [(1, 1), (1, 2), (2, 2), (1, 3)],
        'tfidf__use_idf': [True, False],
        'tfidf__norm': ['l1', 'l2'],
    }


def naive_bayes_gs_opts(dry_run: bool = False):
    return {
        'estimator': Pipeline(_SHARED_PIPELINE_ELEMENTS + [
            ('nb', MultinomialNB())
        ]),
        'param_grid': {
            'nb__alpha': [1e-2, 3e-2, 1e-1, 3e-1, 1],
            **_get_shared_param_grid_opts(dry_run)
        },
        **_get_shared_gs_opts(dry_run)
    }


def logistic_regression_gs_opts(dry_run: bool = False):
    return {
        'estimator': Pipeline(_SHARED_PIPELINE_ELEMENTS + [
            ('lr', LogisticRegression(
                solver='liblinear',
                dual=False
            ))
        ]),
        'param_grid': {
            'lr__C': [1e-2, 3e-2, 0.2, 0.4, 0.6, 0.8, 1, 3, 10, 30],
            'lr__penalty': ['l1', 'l2'],
            **_get_shared_param_grid_opts(dry_run)
        },
        **_get_shared_gs_opts(dry_run)
    }
