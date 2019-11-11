#!/usr/bin/python3


import plac


MODELS = [
'BernoulliNB',
'KNN',
'LinearSVC_L1',
'LinearSVC_L2',
'MultinomialNB',
'NearestCentroid',
'PassiveAggresive',
'Perceptron',
'Ridge',
'SGDClassifierElastic',
'SGDClassifierL2',
]


@plac.annotations(
    event_sel=("Events to select", "positional", None, int),
)
def main(*event_sel):
    for m in MODELS:
        model = __import__(m)
        acc = model.main(event_sel)
        print('>>> {} acc={:.3f}'.format(m, acc))


if __name__ == "__main__":
    plac.call(main)
