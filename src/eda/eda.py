#!/usr/bin/python3

"""

"""


def read_train():
    import pandas as pd

    df = pd.read_csv('../../data/train.csv')
    df.drop_duplicates(subset='text', inplace=True)
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

    print(80 * '-')
    df['strlen'] = df.text.map(len)
    df['nwords'] = df.text.map(lambda s: len(s.split()))

    print('Max string length: {}'.format(df.strlen.max()))
    print('Min string length: {}'.format(df.strlen.min()))
    print('Mean string length: {:.1f}'.format(df.strlen.mean()))
    print('Median string length: {:.1f}'.format(df.strlen.median()))
    print('Mean # of words: {:.1f}'.format(df.nwords.mean()))
    print('Median # of words: {}'.format(df.nwords.median()))
    print('Min # of words: {}'.format(df.nwords.min()))
    print('Max # of words: {}'.format(df.nwords.max()))
    print(80 * '-')

    print('[6x] Max string length: {}'.format(df[df['event'].isin(range(60, 70))].strlen.max()))
    print('[6x] Min string length: {}'.format(df[df['event'].isin(range(60, 70))].strlen.min()))
    print('[6x] Mean string length: {:.1f}'.format(df[df['event'].isin(range(60, 70))].strlen.mean()))
    print('[6x] Median string length: {}'.format(df[df['event'].isin(range(60, 70))].strlen.median()))
    print('[6x] Mean # of words: {:.1f}'.format(df[df['event'].isin(range(60, 70))].nwords.mean()))
    print('[6x] Median # of words: {}'.format(df[df['event'].isin(range(60, 70))].nwords.median()))
    print(80 * '-')

    print('[60] Max string length: {}'.format(df[df['event'] == 60].strlen.max()))
    print('[60] Min string length: {}'.format(df[df['event'] == 60].strlen.min()))
    print('[60] Mean string length: {:.1f}'.format(df[df['event'] == 60].strlen.mean()))
    print('[60] Median string length: {}'.format(df[df['event'] == 60].strlen.median()))
    print('[60] Mean # of words: {:.1f}'.format(df[df['event'] == 60].nwords.mean()))
    print('[60] Median # of words: {}'.format(df[df['event'] == 60].nwords.median()))
    print(80 * '-')

    print('[62] Max string length: {}'.format(df[df['event'] == 62].strlen.max()))
    print('[62] Min string length: {}'.format(df[df['event'] == 62].strlen.min()))
    print('[62] Mean string length: {:.1f}'.format(df[df['event'] == 62].strlen.mean()))
    print('[62] Median string length: {}'.format(df[df['event'] == 62].strlen.median()))
    print('[62] Mean # of words: {:.1f}'.format(df[df['event'] == 62].nwords.mean()))
    print('[62] Median # of words: {}'.format(df[df['event'] == 62].nwords.median()))
    print(80 * '-')

    print('[63] Max string length: {}'.format(df[df['event'] == 63].strlen.max()))
    print('[63] Min string length: {}'.format(df[df['event'] == 63].strlen.min()))
    print('[63] Mean string length: {:.1f}'.format(df[df['event'] == 63].strlen.mean()))
    print('[63] Median string length: {}'.format(df[df['event'] == 63].strlen.median()))
    print('[63] Mean # of words: {:.1f}'.format(df[df['event'] == 63].nwords.mean()))
    print('[63] Median # of words: {}'.format(df[df['event'] == 63].nwords.median()))


    return


if __name__ == "__main__":
    main()
