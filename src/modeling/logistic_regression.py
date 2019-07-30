import sys
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

ROOT_FOLDERPATH = Path.cwd().parent.parent
sys.path.append(str(ROOT_FOLDERPATH))
from src.preparation.load_data import load_dataset

MODEL_FILEPATH = ROOT_FOLDERPATH / 'model' / 'logistic-regression.joblib'


def train():
    dataset = load_dataset()
    pipeline = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2))),
                         ('tfidf', TfidfTransformer()),
                         ('clf', MultinomialNB())])
    param_grid = {
        'tfidf__use_idf': [False],
        'tfidf__norm': ['l1'],
        'clf__alpha': [1e-1],
    }
    gs_options = {
        'estimator': pipeline,
        'param_grid': param_grid,
        'cv': 2,
        'scoring': 'f1_macro',
        'verbose': 5,
        'n_jobs': -1,
    }
    gs = GridSearchCV(**gs_options)
    gs.fit(dataset.train.x, dataset.train.y)
    return gs


def get_model():
    if MODEL_FILEPATH.exists():
        with MODEL_FILEPATH.open('rb') as file:
            classifier = joblib.load(file)
            return classifier

    classifier = train()
    with MODEL_FILEPATH.open('wb') as file:
        joblib.dump(classifier, file)
    return classifier
