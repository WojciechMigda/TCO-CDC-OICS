#!/usr/bin/python3

from ReadData import read_train, read_test

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import LinearSVC

import numpy as np

import plac


class ItemSelector(BaseEstimator, TransformerMixin):
    """For data grouped by feature, select subset of data at a provided key.

    The data is expected to be stored in a 2D data structure, where the first
    index is over features and the second is over samples.  i.e.

    >> len(data[key]) == n_samples

    Please note that this is the opposite convention to scikit-learn feature
    matrixes (where the first index corresponds to sample).

    ItemSelector only requires that the collection implement getitem
    (data[key]).  Examples include: a dict of lists, 2D numpy array, Pandas
    DataFrame, numpy record array, etc.

    >> data = {'a': [1, 5, 2, 5, 2, 8],
               'b': [9, 4, 1, 4, 1, 3]}
    >> ds = ItemSelector(key='a')
    >> data['a'] == ds.transform(data)

    ItemSelector is not designed to handle data grouped by sample.  (e.g. a
    list of dicts).  If your data is structured this way, consider a
    transformer along the lines of `sklearn.feature_extraction.DictVectorizer`.

    Parameters
    ----------
    key : hashable, required
        The key corresponding to the desired value in a mappable.
    """
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]



@plac.annotations(
    kbest=('Number of best features to select with SelectKBest/Chi2', 'option', 'K', int),
    ngram_hi=('Max length of n-grams to generate, (2, G)', 'option', 'G', int),
    jobs=('Model: number of jobs', 'option', 'j', int),
    seed=('Model: RNG seed value', 'option', 's', int),
)
def main(
    kbest=46000,
    ngram_hi=3,
    jobs=1,
    seed=1,
    ):

    print(locals())

    df_tr = read_train()
    df_te = read_test()

    # Przyklady pipeline
    # https://github.com/sfu-natlang/typed-align/blob/3fdab03765524a93831b31c7f21f3e8a11dfe54d/srcs/LogisticRegression.py
    # https://scikit-learn.org/0.18/auto_examples/hetero_feature_union.html

    clf = Pipeline([
        ('union',
            FeatureUnion([

                # vectorized text
                ('vect',
                    Pipeline([
                        ('selector', ItemSelector(key='text')),
                        ('whitespace_remover', FunctionTransformer(lambda col: col.apply(lambda s: ''.join(s.split())), validate=False)),
                        ('digits_remover', FunctionTransformer(lambda col: col.apply(lambda s: ''.join(i for i in s if not i.isdigit())), validate=False)),
                        ('vec', CountVectorizer(binary=True, ngram_range=(2, ngram_hi), analyzer='char', dtype=np.uint8)),
                    ])
                ),

                # sex
                ('sex',
                    Pipeline([
                        ('selector', ItemSelector(key='sex')),
                        ('less_1', FunctionTransformer(lambda col: col.values[:, None] - 1, validate=False)),
                    ])
                ),
            ])
        ),
        ('kbest', SelectKBest(chi2, k=kbest)),
        ('classifier', LinearSVC(loss='squared_hinge', penalty='l2', dual=False, tol=1e-4, verbose=0, random_state=1,
                    C=0.007283075704523443))
    ])

#    foo = clf.fit_transform(df_tr, df_tr.event)
    clf.fit(df_tr, df_tr.event)
#    foo = clf.transform(df_te)
    yhat = clf.predict(df_te)
#    print(foo.shape)
#    print(foo.dtype)
#    print(foo)

    df_te['event'] = yhat

    df_te.to_csv('solution.csv', index=False)



if __name__ == "__main__":
    plac.call(main)
