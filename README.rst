Intro
=====

This is an attempt to solve CDC-OIICS injury event classification TopCoder contest.

Even though authors of the problem suggested improving client's BeRT baseline, due to scarcity of my time I decided to see how high score I could achieve with simple classical ML models instead.

Evaluations I performed allowed me to conclude that LinearSVC classifier from `scikit-learn` applied to *n*-gram text representation performs best in terms of weighted F1 metric and time needed to train and evaluate the model.

With such model my internal CV experiments as well as provisional leaderboard indicated that such simple statistical model scores around 0.846, not far from 0.86-ish client's BeRT model. At the moment of writing this highest score on the provisional leaderboard scored around 0.895.

*Edited with retext*