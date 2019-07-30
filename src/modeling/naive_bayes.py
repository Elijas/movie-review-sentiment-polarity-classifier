from pprint import pprint

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from src.preparation.load_data import load_dataset


def preprocess_and_train():
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
    gs_result = gs.fit(dataset.train.x, dataset.train.y)

    print(repr(gs))
    print(classification_report(dataset.test.y, gs.predict(dataset.test.x), digits=3))

    test_accuracy = gs.score(dataset.test.x, dataset.test.y)
    print('Best Accuracy : '
          '{:.4f}\n{}\nTest Accuracy : {:.4f}\n\n'.format(gs_result.best_score_,
                                                          gs_result.best_params_,
                                                          test_accuracy))


if __name__ == '__main__':
    pprint(preprocess_and_train())
