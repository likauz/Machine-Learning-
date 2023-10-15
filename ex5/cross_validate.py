from __future__ import annotations

import math
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
import pandas as pd

from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """

    start = 0
    part = 0
    score = []
    valid_score = []
    split_fractions = np.ones((cv, 1)) * (1/cv)
    for fraction in split_fractions:
        part = math.ceil(fraction * X.shape[0]) + part
        train_x = np.concatenate([X[:start], X[part:]], axis=0)
        train_y = np.concatenate([y[:start], y[part:]], axis=0)
        test_x = X[start: part]
        test_y = y[start: part]
        start = part
        estimator.fit(train_x, train_y)
        train_predict = estimator.predict(train_x)
        score.append(scoring(train_predict, train_y))
        if test_y.shape[0] > 0:
            test_predict = estimator.predict(test_x)
            valid_score.append(scoring(test_predict, test_y))
    return (1 / cv) * sum(score), (1 / cv) * sum(valid_score)








