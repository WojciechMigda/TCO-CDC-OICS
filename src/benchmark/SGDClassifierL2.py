#!/usr/bin/python3
# -*- coding: utf-8 -*-


from ReadData import read_train
from Benchmark import benchmark
from Preprocess import preprocess


def main(event_sel=None):
    df = read_train()

#    X_train, y_train = preprocess(df, event_sel=[71, 62, 42])
#    X_train, y_train = preprocess(df, event_sel=[62, 63, 60])
    X_train, y_train = preprocess(df, event_sel=event_sel)

    from sklearn.linear_model import SGDClassifier
    clf = SGDClassifier(alpha=.0001, max_iter=50, tol=1e-3, penalty='l2', random_state=1)

    return benchmark(clf, X_train, y_train)

if __name__ == "__main__":
    main()
