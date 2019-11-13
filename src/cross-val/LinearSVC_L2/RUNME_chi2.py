#!/usr/bin/python3

from ReadData import read_train
from Preprocess import preprocess

import numpy as np
from sklearn.metrics import f1_score

import plac


def scoring_w_f1(clf, X, y):
    yhat = clf.predict(X)
    rv = f1_score(y, yhat, average='weighted')
    return rv


def evaluate_hyper(train_X, train_y, objective,
                   neval, nfolds, ncvjobs, nreps,
                   nbest,
                   njobs, seed,
                   ):
    from hyperopt import fmin, tpe, hp

    space = {
        'C': hp.loguniform('x_C', np.log(0.003), np.log(0.015)),
        'KBest': hp.quniform ('x_KBest', 34000, 48000, 1000),
        }

    from functools import partial
    objective_xy = partial(objective, train_X, train_y,
                           nfolds, ncvjobs, nreps,
                           nbest,
                           seed, njobs)

    best = fmin(fn=objective_xy,
            space=space,
            algo=tpe.suggest,
            max_evals=neval,
            )
    return best


def hyper_objective(train_X, train_y, nfolds, ncvjobs, nreps,
                    nbest,
                    seed, n_jobs,
                    space):
    kwargs = {}
    for k, v in space.items():
        if k in ['KBest']:
            v = int(v)
            pass
        kwargs[k] = v
        pass

    from sklearn.pipeline import Pipeline
    from sklearn.feature_selection import SelectKBest, chi2
    from sklearn.svm import LinearSVC

    if 'KBest' in kwargs:
        nbest = kwargs['KBest']
        del kwargs['KBest']
    else:
        nbest = nbest or train_X.shape[1]

    clf = Pipeline([
        ('kbest', SelectKBest(chi2, k=nbest)),
        ('train', LinearSVC(loss='squared_hinge', penalty='l2', dual=False, tol=1e-3, verbose=0, random_state=seed,
                    **kwargs))
    ])

    from sklearn.model_selection import StratifiedKFold

    scores = []

    from sklearn.model_selection import cross_val_score
    for i in range(nreps):
        kf = StratifiedKFold(n_splits=nfolds, random_state=seed + i, shuffle=True)
        scores.extend(cross_val_score(clf, train_X, train_y, cv=kf, n_jobs=ncvjobs, scoring=scoring_w_f1))

    score = np.mean(scores)
    std = np.std(scores)

    kwargs['KBest'] = nbest
    print('best score: {:.5f} +/- {:.5f}  best params: {}'.format(score, std, kwargs))
    return -score



@plac.annotations(
    event_sel=("Events to select", "positional", None, int),
    neval=('Number of hyper-parameter sets evaluations', 'option', 'N', int),
    nfolds=('Number of cross-validation folds', 'option', 'F', int),
    ncvjobs=('Number of cross-validation (cross_val_score) jobs', 'option', 'cv-jobs', int),
    nreps=('Number of cross-validation repetitions (how many times cross_val_score will be called for the same parameters)', 'option', 'R', int),
    kbest=('Number of best features to select with SelectKBest/Chi2', 'option', 'K', int),
    ngram_hi=('Max length of n-grams to generate, (2, G)', 'option', 'G', int),
    jobs=('Model: number of jobs', 'option', 'j', int),
    seed=('Model: RNG seed value', 'option', 's', int),
)
def main(
    neval=30,
    nfolds=5,
    ncvjobs=1,
    nreps=5,
    kbest=None,
    ngram_hi=3,
    jobs=1,
    seed=1,
    *event_sel
):
    print(locals())
    #return
    #event_sel=[70, 71]
    #nreps=1
    df = read_train()
    X, y = preprocess(df, event_sel=event_sel, ngrams=(2, ngram_hi))

    best = evaluate_hyper(X, y, hyper_objective,
                          neval=neval, nfolds=nfolds, ncvjobs=ncvjobs, nreps=nreps,
                          nbest=kbest,
                          njobs=jobs, seed=seed,
                          )

    print('Final best: {}'.format(best))


    return


if __name__ == "__main__":
    plac.call(main)

# 71 62 42 55 63 60 11 73 43 70 64 53 13 66 26 12 41 99 24 31 78 27 72 51 52 44 32 23 25 69 67 61 22 40 65 21 50 49 45 79 20 54 56
