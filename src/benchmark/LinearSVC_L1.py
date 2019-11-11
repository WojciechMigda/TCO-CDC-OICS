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

    from sklearn.svm import LinearSVC
    clf = LinearSVC(loss='squared_hinge', penalty='l1', dual=False, tol=1e-3, max_iter=2000, verbose=0, random_state=1)

    return benchmark(clf, X_train, y_train)

if __name__ == "__main__":
    main()
