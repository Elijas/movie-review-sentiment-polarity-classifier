import sys
from pathlib import Path

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from src.evaluation.evaluate_classifier_performance import evaluate_classifier_performance
from src.modeling._model_file_io import load_model, save_model

ROOT_FOLDERPATH = Path.cwd().parent.parent
sys.path.append(str(ROOT_FOLDERPATH))
from src.preparation.load_data import load_dataset

MODEL_FILEPATH = ROOT_FOLDERPATH / 'model' / 'naive-bayes.joblib'


def train(dataset=None):
    dataset = dataset or load_dataset()
    pipeline = Pipeline([('vect', CountVectorizer(ngram_range=(1, 3))),
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
        'scoring': 'accuracy',
        'verbose': 5,
        'n_jobs': -1,
    }
    gs = GridSearchCV(**gs_options)
    gs.fit(dataset.train.x, dataset.train.y)
    return gs


def get_model():
    if MODEL_FILEPATH.exists():
        return load_model(MODEL_FILEPATH)
    classifier = train()
    save_model(MODEL_FILEPATH, classifier)
    return classifier


if __name__ == '__main__':
    dataset = load_dataset()
    classifier = train(dataset)
    save_model(MODEL_FILEPATH, classifier)
    evaluate_classifier_performance(classifier, dataset)
