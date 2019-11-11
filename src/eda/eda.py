#!/usr/bin/python3

"""

"""


def read_train():
    import pandas as pd

    df = pd.read_csv('../../data/train.csv')
    print('Full TRAIN shape: {}'.format(df.shape))

    return df


def main():
    import pandas as pd

    df = read_train()

#    TRAIN['text'] = TRAIN['text'].apply(lambda s: ''.join(s.split()))
#    TRAIN['text'] = TRAIN['text'].apply(lambda s: ''.join(i for i in s if not i.isdigit()))
#    print(TRAIN.head())

    gb = df.groupby('event')
    print(gb.count()['text'].sort_values(ascending=False))

    #gb = df.groupby('age')
    #print(gb.count()['text'].sort_values(ascending=False))
    print('Min age {}'.format(df.age.min()))
    print('Max age {}'.format(df.age.max()))

    # top-3 same group: 62, 63, 60
    # top-3 diff group: 71, 62, 42

    print(df[df['event'].isin([71, 62, 42])].head())
    print(df[df['event'].isin([62, 63, 60])].head())

    return


if __name__ == "__main__":
    main()
