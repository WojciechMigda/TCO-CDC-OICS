#!/usr/bin/python3

from ReadData import read_train
from Preprocess import preprocess

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
    import numpy as np

    space = {
        'KBest': hp.quniform ('x_KBest', 12000, 24000, 2000),
        'n_estimators': hp.quniform ('x_n_estimators', 200, 1000, 200),
        #'boosting_type': hp.choice('x_boosting_type', ['gbdt', 'dart', 'goss']),  # no 'rf', causes crash
        'boosting_type': hp.choice('x_boosting_type', ['gbdt']),  # no 'rf', causes crash
        'num_leaves': hp.quniform('num_leaves', 8, 128, 2),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1.0),
        'max_depth': hp.quniform('max_depth', 1, 10, 1),
        "min_child_weight": hp.uniform('min_child_weight', 0.5, 10),

        "learning_rate": hp.uniform("learning_rate", 0.01, 0.05),
        #"learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.2)),

        # subsample (float, optional (default=1.)) – Subsample ratio of the training instance.

        #'min_data_in_leaf':  hp.quniform ('min_data_in_leaf', 10, 200, 1),
        #'feature_fraction': hp.uniform('feature_fraction', 0.75, 1.0),
        #'bagging_fraction': hp.uniform('bagging_fraction', 0.75, 1.0),

        #"feature_fraction": hp.quniform("feature_fraction", 0.6, 1.0, 0.1),
        #"bagging_fraction": hp.quniform("bagging_fraction", 0.6, 1.0, 0.1),
        #"bagging_freq": hp.choice("bagging_freq", np.linspace(0, 50, 10, dtype=int)),
        #"reg_alpha": hp.uniform("reg_alpha", 0, 2),
        #"reg_lambda": hp.uniform("reg_lambda", 0, 2),

        #"feature_fraction": hp.quniform("feature_fraction", 0.5, 1.0, 0.1),
        #"bagging_fraction": hp.quniform("bagging_fraction", 0.5, 1.0, 0.1),
        #"bagging_freq": hp.choice("bagging_freq", np.linspace(0, 50, 10, dtype=int)),
        #"reg_alpha": hp.uniform("reg_alpha", 0, 30),
        #"reg_lambda": hp.uniform("reg_lambda", 0, 30),

        #'min_sum_hessian_in_leaf': hp.loguniform('min_sum_hessian_in_leaf', 0, 2.3),
        #'max_bin': hp.quniform ('max_bin', 64, 512, 1),
        #'bagging_freq': hp.quniform ('bagging_freq', 1, 5, 1),
        #'lambda_l1': hp.uniform('lambda_l1', 0, 10 ),
        #'lambda_l2': hp.uniform('lambda_l2', 0, 10 ),

        #'num_leaves': hp.quniform('num_leaves', 50, 500, 50),
        #'bagging_fraction': hp.quniform('bagging_fraction', 0.5, 0.9, 0.05),
        #'feature_fraction': hp.quniform('feature_fraction', 0.25, 0.55, 0.05),
        #'min_data_in_leaf': hp.quniform('min_data_in_leaf', 100, 500, 50),
        #'lambda_l1':  hp.loguniform('lambda_l1', -3, 2),
        #'lambda_l2': hp.loguniform('lambda_l2', -3, 2),
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
        if k in ['KBest', 'n_estimators', 'num_leaves', 'max_depth', 'min_data_in_leaf', 'max_bin']:
            v = int(v)
            pass
        kwargs[k] = v
        pass

    from sklearn.pipeline import Pipeline
    from sklearn.feature_selection import SelectKBest, chi2, f_classif
    from lightgbm import LGBMClassifier

    if 'KBest' in kwargs:
        nbest = kwargs['KBest']
        del kwargs['KBest']
    else:
        nbest = nbest or train_X.shape[1]

    clf = Pipeline([
        ('kbest', SelectKBest(chi2, k=nbest)),
        ('train', LGBMClassifier(random_state=seed, verbose=-1, silent=1, **kwargs))
    ])

    from sklearn.model_selection import StratifiedKFold

    scores = []

    from sklearn.model_selection import cross_val_score
    for i in range(nreps):
        kf = StratifiedKFold(n_splits=nfolds, random_state=seed + i, shuffle=True)
        scores.extend(cross_val_score(clf, train_X, train_y, cv=kf, n_jobs=ncvjobs, scoring=scoring_w_f1))

    import numpy as np
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
    *event_sel,
):
    print(locals())
    #return
    #event_sel=[70, 71]
    #nreps=1
    df = read_train()
    X, y = preprocess(df, event_sel=event_sel, ngrams=(2, ngram_hi))

    best = evaluate_hyper(X.astype(float), y, hyper_objective,
                          neval=neval, nfolds=nfolds, ncvjobs=ncvjobs, nreps=nreps,
                          nbest=kbest,
                          njobs=jobs, seed=seed,
                          )

    print('Final best: {}'.format(best))


    return


if __name__ == "__main__":
    plac.call(main)

# 71 62 42 55 63 60 11 73 43 70 64 53 13 66 26 12 41 99 24 31 78 27 72 51 52 44 32 23 25 69 67 61 22 40 65 21 50 49 45 79 20 54 56
