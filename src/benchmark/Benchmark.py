# -*- coding: utf-8 -*-

import time

def scoring_w_f1(y, yhat):
    from sklearn.metrics import f1_score
    rv = f1_score(y, yhat, average='weighted')
    return rv


def benchmark(clf, X_train, y_train, n_iter=None):
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time.time()
    if n_iter:
        clf.fit(X_train, y_train, n_iter)
    else:
        clf.fit(X_train, y_train)
    train_time = time.time() - t0
    print("train time: %0.3fs" % train_time)
    #acc = clf.score(X_train, y_train) * 100.
    acc = scoring_w_f1(y_train, clf.predict(X_train)) * 100.
    print('Accuracy on Train: {:.3f}'.format(acc))
    return acc
