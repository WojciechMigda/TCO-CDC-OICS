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
                   njobs, seed,
                   ):
    from hyperopt import fmin, tpe, hp

    space = {
#        'C': hp.choice("x_X", [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]),
#        'C': hp.choice("x_X", [0.003, 0.01, 0.03, 0.1, 0.3, 1]),
        'C_exp': hp.quniform ('x_C_exp', -4, -1, 0.2),
#        'boost_true_positive_feedback': hp.choice("x_boost_true_positive_feedback", [0, 1]),
#        'number_of_states': hp.quniform("x_number_of_states", states_min, states_max, states_step),
#        'threshold': hp.quniform ('x_threshold', threshold_min, threshold_max, threshold_step),
#        's': hp.uniform ('x_s', s_min, s_max),
        }

    from functools import partial
    objective_xy = partial(objective, train_X, train_y,
                           nfolds, ncvjobs, nreps,
                           seed, njobs)

    best = fmin(fn=objective_xy,
            space=space,
            algo=tpe.suggest,
            max_evals=neval,
            )
    return best


def hyper_objective(train_X, train_y, nfolds, ncvjobs, nreps,
                    seed, n_jobs,
                    space):
    kwargs = {}
    for k, v in space.items():
        if k in ['boost_true_positive_feedback', 'number_of_states',
                 'threshold']:
            v = int(v)
            pass
        kwargs[k] = v
        pass

    kwargs['C'] = 10 ** kwargs['C_exp']
    del kwargs['C_exp']
    #print('DEBUG: C={:.5f}'.format(kwargs['C']))

    from sklearn.svm import LinearSVC
    clf = LinearSVC(loss='squared_hinge', penalty='l2', dual=False, tol=1e-3, verbose=0, random_state=seed,
                    **kwargs)

    from sklearn.model_selection import StratifiedKFold

    scores = []

    from sklearn.model_selection import cross_val_score
    for i in range(nreps):
        kf = StratifiedKFold(n_splits=nfolds, random_state=seed + i, shuffle=True)
        scores.extend(cross_val_score(clf, train_X, train_y, cv=kf, n_jobs=ncvjobs, scoring=scoring_w_f1))

    import numpy as np
    score = np.mean(scores)
    std = np.std(scores)

    print('best score: {:.5f} +/- {:.5f}  best params: {}'.format(score, std, kwargs))
    return -score



@plac.annotations(
    event_sel=("Events to select", "positional", None, int),
    neval=('Number of hyper-parameter sets evaluations', 'option', 'N', int),
    nfolds=('Number of cross-validation folds', 'option', 'F', int),
    ncvjobs=('Number of cross-validation (cross_val_score) jobs', 'option', 'cv-jobs', int),
    nreps=('Number of cross-validation repetitions (how many times cross_val_score will be called for the same parameters)', 'option', 'R', int),
    jobs=('Model: number of jobs', 'option', 'j', int),
    seed=('Model: RNG seed value', 'option', 's', int),
)
def main(
    neval=30,
    nfolds=5,
    ncvjobs=1,
    nreps=5,
    jobs=1,
    seed=1,
    *event_sel,
):
    print(locals())
    #return
    #event_sel=[70, 71]
    #nreps=1
    df = read_train()
    X, y = preprocess(df, event_sel=event_sel)

    best = evaluate_hyper(X, y, hyper_objective,
                          neval=neval, nfolds=nfolds, ncvjobs=ncvjobs, nreps=nreps,
                          njobs=jobs, seed=seed,
                          )

    print('Final best: {}'.format(best))


    return


if __name__ == "__main__":
    plac.call(main)

# 71 62 42 55 63 60 11 73 43 70 64 53 13 66 26 12 41 99 24 31 78 27 72 51 52 44 32 23 25 69 67 61 22 40 65 21 50 49 45 79 20 54 56
