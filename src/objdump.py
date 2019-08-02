import joblib


def dump(obj, path):
    with path.open('wb') as file:
        joblib.dump(obj, file)


def load(path):
    with path.open('rb') as file:
        return joblib.load(file)
