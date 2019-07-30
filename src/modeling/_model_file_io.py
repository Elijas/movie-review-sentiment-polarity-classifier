import joblib


def save_model(path, classifier):
    with path.open('wb') as file:
        joblib.dump(classifier, file)


def load_model(path):
    with path.open('rb') as file:
        classifier = joblib.load(file)
        return classifier
