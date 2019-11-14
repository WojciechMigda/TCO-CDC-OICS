# -*- coding: utf-8 -*-


def read_train():
    import pandas as pd

    df = pd.read_csv('../../../data/train.csv')
    print('Removing duplicates')
    df.drop_duplicates(subset='text', inplace=True)

    print('Full TRAIN shape: {}'.format(df.shape))

    return df

def read_test():
    import pandas as pd

    df = pd.read_csv('../../../data/test.csv')

    print('Full TEST shape: {}'.format(df.shape))

    return df
