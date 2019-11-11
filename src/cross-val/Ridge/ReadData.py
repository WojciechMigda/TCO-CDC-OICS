# -*- coding: utf-8 -*-


def read_train():
    import pandas as pd

    df = pd.read_csv('../../../data/train.csv')
    print('Full TRAIN shape: {}'.format(df.shape))

    return df
