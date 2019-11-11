# -*- coding: utf-8 -*-

def preprocess(df,
               event_sel=None, strip_whitespace=True, strip_digits=True,
               ngrams=(2, 3)):

    if event_sel:
        df = df[df['event'].isin(event_sel)].copy()
        #df = df[df['event'].isin([71, 62, 42])]
        #df = df[df['event'].isin([62, 63, 60])]

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

    print(X_train.shape)
    print(X_train.dtype)

    """
    Bez spacji
    3-grams z cyframi       19351
    3-grams bez cyfr        11818
    2-, 3-grams bez cyfr    12481
    """
    X_train.sort_indices()
    print('TRAIN indices sorted')
    print(X_train.shape)

    return X_train, df['event']
