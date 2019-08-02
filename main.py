import sys

from sklearn.model_selection import GridSearchCV

from src.evaluation import get_classification_report
from src.io import get_raw_structured_dataset, save_model, Dataset
from src.processing import only_first_few_samples
from src.training import naive_bayes_gs_opts, logistic_regression_gs_opts


def get_data(dry_run: bool = False):
    data = get_raw_structured_dataset()
    return data if not dry_run else only_first_few_samples(data)


def train_classifier(gs_options: dict, dataset: Dataset):
    classifier = GridSearchCV(**gs_options)
    return classifier.fit(dataset.trn.x, dataset.trn.y)


def show_and_save_results(classifier_name: str, classifier, dataset: Dataset) -> None:
    report = get_classification_report(classifier, dataset)
    print(report)
    save_model(classifier_name, classifier, report)


def run_training_session(dry_run: bool = False):
    data = get_data(dry_run)

    logistic_regression_classifier = train_classifier(logistic_regression_gs_opts(dry_run), data)
    show_and_save_results('logistic-regression', logistic_regression_classifier, data)

    naive_bayes_classifier = train_classifier(naive_bayes_gs_opts(dry_run), data)
    show_and_save_results('naive-bayes', naive_bayes_classifier, data)


if __name__ == '__main__':
    run_training_session(dry_run='--dry-run' in sys.argv)
