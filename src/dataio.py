from collections import namedtuple

from sklearn.model_selection import train_test_split

from src import const, objdump

Corpus = namedtuple('Corpus', ['pos', 'neg', 'neg_and_pos'])
XY = namedtuple('XY', 'x y')
Dataset = namedtuple('Dataset', 'trn tst')


def get_corpus():
    with const.PATHS.CORPUS_NEG.open() as file:
        neg = file.readlines()
    with const.PATHS.CORPUS_POS.open() as file:
        pos = file.readlines()
    return Corpus(neg=neg, pos=pos, neg_and_pos=neg + pos)


def get_split_dataset():
    if (const.PATHS.SPLIT_DATASET_TRN.exists()
            and const.PATHS.SPLIT_DATASET_TST.exists()):
        # Saving the split dataset to ensure an appropriate test dataset
        #  for the trained model that has been saved
        return Dataset(trn=objdump.load(const.PATHS.SPLIT_DATASET_TRN),
                       tst=objdump.load(const.PATHS.SPLIT_DATASET_TST))

    def corpus_to_dataset(*args, **kwargs):
        trn_x, tst_x, trn_y, tst_y = train_test_split(*args, **kwargs)
        return Dataset(trn=XY(x=trn_x, y=trn_y),
                       tst=XY(x=tst_x, y=tst_y))

    corpus = get_corpus()
    labels = ([const.LABELS.NEG] * len(corpus.neg) + [const.LABELS.POS] * len(corpus.pos))
    dataset = corpus_to_dataset(corpus.neg_and_pos, labels,
                                test_size=const.DATASET_TEST_SPLIT_RATIO,
                                random_state=const.RANDOMNESS_SEED)

    objdump.dump(dataset.tst, const.PATHS.SPLIT_DATASET_TST)
    objdump.dump(dataset.trn, const.PATHS.SPLIT_DATASET_TRN)
    return dataset


def get_label_title(label):
    return const.LABELS.TITLES[label]
