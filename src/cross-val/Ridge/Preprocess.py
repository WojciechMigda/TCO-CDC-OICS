# -*- coding: utf-8 -*-


def OneHot(col, colname):
    from pandas import get_dummies

    dummies = get_dummies(col)
    dummies.rename(columns={p: colname + '_' + str(i + 1)  for i, p in enumerate(dummies.columns.values)}, inplace=True)
    return dummies


def preprocess(df,
               event_sel=None, strip_whitespace=True, strip_digits=True,
               ngrams=(2, 3)):

    if event_sel:
        df = df[df['event'].isin(event_sel)].copy()

    if strip_whitespace:
        print('Removing whitespace.')
        df['text'] = df['text'].apply(lambda s: ''.join(s.split()))
    if strip_digits:
        print('Removing digits.')
        df['text'] = df['text'].apply(lambda s: ''.join(i for i in s if not i.isdigit()))

    print(df.head())

    import numpy as np
    from sklearn.feature_extraction.text import CountVectorizer

    vectorizer = CountVectorizer(binary=True, ngram_range=ngrams, analyzer='char', dtype=np.uint8)
    print('Vectorizing with {}'.format(vectorizer))

    X_train = vectorizer.fit_transform(df['text'])

    print('TRAIN corpus vectorized')
    from scipy import sparse
    X_train = sparse.hstack((X_train, df.sex.values[:, None] - 1), dtype='uint8')

    """
    one_hot_age = OneHot(df.age // 5, 'age').values
    X_train = sparse.hstack((X_train, one_hot_age), dtype='uint8')
    """

    print(X_train.dtype)

    """
    Bez spacji
    3-grams z cyframi       19351
    3-grams bez cyfr        11818
    2-, 3-grams bez cyfr    12481
    """
    #X_train.sort_indices()
    #print('TRAIN indices sorted')
    print(X_train.shape)

    return X_train, df['event'].values
