#!/usr/bin/python3


import numpy as np
import pandas as pd
import plac

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2

from ItemSelector import ItemSelector


def whitespace_remover(col):
    return col.apply(lambda s: ''.join(s.split()))


def digits_remover(col):
    return col.apply(lambda s: ''.join(i for i in s if not i.isdigit()))


def decrementer(col):
    return col.values[:, None] - 1


def train(df, y, seed, kbest, C, ngram_hi):
    clf = Pipeline([
        ('union',
            FeatureUnion([

                # vectorized text
                ('vect',
                    Pipeline([
                        ('selector', ItemSelector(key='text')),
                        ('whitespace_remover', FunctionTransformer(whitespace_remover, validate=False)),
                        ('digits_remover', FunctionTransformer(digits_remover, validate=False)),
                        ('vec', CountVectorizer(binary=True, ngram_range=(2, ngram_hi), analyzer='char', dtype=np.uint8)),
                    ])
                ),

                # sex
                ('sex',
                    Pipeline([
                        ('selector', ItemSelector(key='sex')),
                        ('less_1', FunctionTransformer(decrementer, validate=False)),
                    ])
                ),
            ])
        ),
        ('kbest', SelectKBest(chi2, k=kbest)),
        ('classifier', LinearSVC(loss='squared_hinge', penalty='l2', dual=False, tol=1e-2, verbose=1, random_state=seed, C=C))
    ])
    print(clf)
    print('-' * 80)

    clf.fit(df, y)

    return clf


def train_all(df, seed, ngram_hi=5):

    # train root classes
    print('=' * 80)
    print('Training root event classifier')
    print('-' * 80)
    root_clf = train(df, df.event // 10, seed=seed, kbest=44000, C=0.010804553291322362, ngram_hi=ngram_hi)
    print('Root event classifier trained.\n')

    # train 70s class
    print('=' * 80)
    print('Training 70s event classifier')
    print('-' * 80)
    df70 = df[df.event.isin(range(70, 80))]
    e70_clf = train(df70, df70.event, seed=seed, kbest=45000, C=0.004086730962398147, ngram_hi=ngram_hi)
    print('70s event classifier trained.\n')

    # train 60s class
    print('=' * 80)
    print('Training 60s event classifier')
    print('-' * 80)
    df60 = df[df.event.isin(range(60, 70))]
    e60_clf = train(df60, df60.event, seed=seed, kbest=45000, C=0.00336597136150739, ngram_hi=ngram_hi)
    print('60s event classifier trained.\n')

    # train 40s class
    print('=' * 80)
    print('Training 40s event classifier')
    print('-' * 80)
    df40 = df[df.event.isin(range(40, 50))]
    e40_clf = train(df40, df40.event, seed=seed, kbest=36000, C=0.00793619084782183, ngram_hi=ngram_hi)
    print('40s event classifier trained.\n')

    # train 50s class
    print('=' * 80)
    print('Training 50s event classifier')
    print('-' * 80)
    df50 = df[df.event.isin(range(50, 60))]
    e50_clf = train(df50, df50.event, seed=seed, kbest=36000, C=0.014771206035450795, ngram_hi=ngram_hi)
    print('50s event classifier trained.\n')

    # train 10s class
    # {'event_sel': (11, 13, 12), 'seed': 1, 'jobs': 1, 'ngram_hi': 5, 'kbest': None, 'nreps': 3, 'ncvjobs': 3, 'nfolds': 6, 'neval': 50}
    # best score: 0.90290 +/- 0.00605  best params: {'C': 0.007641519218362063, 'KBest': 27000}
    print('=' * 80)
    print('Training 10s event classifier')
    print('-' * 80)
    df10 = df[df.event.isin(range(10, 20))]
    e10_clf = train(df10, df10.event, seed=seed, kbest=27000, C=0.007641519218362063, ngram_hi=ngram_hi)
    print('10s event classifier trained.\n')

    # train 20s/30s class
    # {'event_sel': (26, 24, 27, 23, 31, 32), 'seed': 1, 'jobs': 1, 'ngram_hi': 5, 'kbest': None, 'nreps': 3, 'ncvjobs': 3, 'nfolds': 6, 'neval': 50}
    # best score: 0.90962 +/- 0.00555  best params: {'C': 0.0025543686062545205, 'KBest': 42000}
    print('=' * 80)
    print('Training 20s/30s event classifier')
    print('-' * 80)
    df23 = df[df.event.isin(range(20, 40))]
    e23_clf = train(df23, df23.event, seed=seed, kbest=42000, C=0.0025543686062545205, ngram_hi=ngram_hi)
    print('20s/30s event classifier trained.\n')

    rv = {
        'clf_root.joblib': root_clf,
        'clf_70.joblib': e70_clf,
        'clf_60.joblib': e60_clf,
        'clf_40.joblib': e40_clf,
        'clf_50.joblib': e50_clf,
        'clf_10.joblib': e10_clf,
        'clf_23.joblib': e23_clf,
    }

    return rv


@plac.annotations(
    ifname=("Input training data CSV file", "positional", None, str),
)
def main(ifname, seed=1):
    print(locals())

    with open(ifname, "rt") as ifile:
        df = pd.read_csv(ifile)
        print('Read input data file <{}>, {}'.format(ifname, df.shape))
        print('Removing \'text\' duplicates')
        df.drop_duplicates(subset='text', inplace=True)
        print(df.head())
        print()

        models = train_all(df, seed)

        for ofname, clf in models.items():
            from sklearn.externals import joblib
            joblib.dump(clf, ofname)

if __name__ == "__main__":
    plac.call(main)
