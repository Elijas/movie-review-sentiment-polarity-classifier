"""
Functions related to saving/loading data and models
"""
import os
from collections import namedtuple
from pathlib import Path

import joblib
from sklearn.model_selection import train_test_split

from src import const

Corpus = namedtuple('Corpus', ['pos', 'neg', 'neg_and_pos'])
XY = namedtuple('XY', 'x y')
Dataset = namedtuple('Dataset', 'trn tst')


def dump(obj, path: Path):
    create_folders_if_needed(path)
    with path.open('wb') as file:
        joblib.dump(obj, file)


def load(path: Path):
    with path.open('rb') as file:
        return joblib.load(file)


def get_corpus():
    with const.PATHS.CORPUS_NEG.open() as file:
        neg = file.readlines()
    with const.PATHS.CORPUS_POS.open() as file:
        pos = file.readlines()
    return Corpus(neg=neg, pos=pos, neg_and_pos=neg + pos)


def get_raw_structured_dataset() -> Dataset:
    """
    :return: Dataset that was shuffled, coupled with labels, split to test/train, but without preprocessing
    """
    if (const.PATHS.RAW_STRUCTURED_DATASET_TRN.exists()
            and const.PATHS.RAW_STRUCTURED_DATASET_TST.exists()):
        # Saving the split dataset to ensure an appropriate test dataset
        #  for the trained model that has been saved
        return Dataset(trn=load(const.PATHS.RAW_STRUCTURED_DATASET_TRN),
                       tst=load(const.PATHS.RAW_STRUCTURED_DATASET_TST))

    def corpus_to_dataset(*args, **kwargs):
        trn_x, tst_x, trn_y, tst_y = train_test_split(*args, **kwargs)
        return Dataset(trn=XY(x=trn_x, y=trn_y),
                       tst=XY(x=tst_x, y=tst_y))

    corpus = get_corpus()
    labels = ([const.LABELS.NEG] * len(corpus.neg) + [const.LABELS.POS] * len(corpus.pos))
    dataset = corpus_to_dataset(corpus.neg_and_pos, labels,
                                test_size=const.DATASET_TEST_SPLIT_RATIO,
                                random_state=const.RANDOMNESS_SEED)

    dump(dataset.tst, const.PATHS.RAW_STRUCTURED_DATASET_TST)
    dump(dataset.trn, const.PATHS.RAW_STRUCTURED_DATASET_TRN)
    return dataset


def get_label_title(label):
    return const.LABELS.TITLES[label]


def create_folders_if_needed(path: Path):
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)


def write_file(contents: str, path: Path):
    create_folders_if_needed(path)
    with path.open('w') as report_file:
        report_file.write(contents)


def save_model(classifier_name: str, classifier, report: str) -> None:
    model_filepath = const.PATHS.MODEL_FOLDER / f'{classifier_name}.joblib'
    report_filepath = const.PATHS.MODEL_FOLDER / f'{classifier_name}.report.txt'

    dump(classifier, model_filepath)
    write_file(report, report_filepath)

    print(f'Successfully saved model and report of classifier "{classifier_name}".')
