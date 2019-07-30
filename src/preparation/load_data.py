from pathlib import Path

ROOT_FOLDERPATH = Path.cwd().parent.parent
DATASET_FOLDERPATH = ROOT_FOLDERPATH / 'data' / 'raw'
NEG_DATASET_FILEPATH = DATASET_FOLDERPATH / 'rt-polarity.neg'
POS_DATASET_FILEPATH = DATASET_FOLDERPATH / 'rt-polarity.pos'


def load_corpus_posneg():
    with NEG_DATASET_FILEPATH.open() as file:
        corpus_neg = file.readlines()

    with POS_DATASET_FILEPATH.open() as file:
        corpus_pos = file.readlines()

    return corpus_pos, corpus_neg
