#!/usr/bin/python3
# -*- coding: utf-8 -*-


from ReadData import read_train
from Benchmark import benchmark
from Preprocess import preprocess


def main(event_sel=None):
    df = read_train()

#    X_train, y_train = preprocess(df, event_sel=event_sel)
#    X_train, y_train = preprocess(df, event_sel=[31, 78])
#    X_train, y_train = preprocess(df, event_sel=[71, 62, 42])
#    X_train, y_train = preprocess(df, event_sel=[71, 62, 42, 55, 11])
    X_train, y_train = preprocess(df, event_sel=[62, 63, 60])

    from lightgbm import LGBMClassifier
    clf = LGBMClassifier(verbose=1, random_state=1, silent=0, n_estimators=400)

    return benchmark(clf, X_train.astype(float), y_train)

if __name__ == "__main__":
    main()
