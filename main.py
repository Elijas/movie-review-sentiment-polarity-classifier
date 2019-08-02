from sklearn.model_selection import GridSearchCV

from src import const
from src.evaluation import get_classification_report
from src.io import get_raw_structured_dataset, save_model, Dataset


def train(gs_options: dict, dataset: Dataset):
    classifier = GridSearchCV(**gs_options)
    return classifier.fit(dataset.trn.x, dataset.trn.y)


def show_and_save_results(classifier_name: str, classifier) -> None:
    report = get_classification_report(classifier, data)
    print(report)
    save_model(classifier_name, classifier, report)


if __name__ == '__main__':
    data = get_raw_structured_dataset()

    naive_bayes_classifier = train(gs_options=const.TRAINING_GS_OPTS.NAIVE_BAYES, dataset=data)
    show_and_save_results(classifier_name='naive-bayes', classifier=naive_bayes_classifier)
