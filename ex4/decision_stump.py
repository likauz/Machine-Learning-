from __future__ import annotations
from typing import Tuple, NoReturn, List, Any
from IMLearn.base import BaseEstimator
import numpy as np
from itertools import product

from IMLearn.metrics import misclassification_error


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """
    def __init__(self):
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        min_error = np.inf

        for sign, feature in product([1, -1], range(X.shape[1])):
            threshold, error = self._find_threshold(X[:, feature], y, sign)
            if min_error > error:
                self.threshold_ = threshold
                min_error = error
                self.sign_ = sign
                self.j_ = feature


    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for


        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        col_threshold = X[:, self.j_]
        y = []
        for val in col_threshold:
            if self.threshold_ <= val:
                y.append(self.sign_)
            else:
                y.append(-self.sign_)
        return np.array(y)

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        sort_values = np.sort(values)
        index_to_sort = np.argsort(values)
        sort_labels = np.take(labels, index_to_sort)
        loss = []
        res = np.ones(len(labels)) * sign
        for i in range(len(sort_values)):
            loss.append(np.sum(np.where(res != np.sign(sort_labels), np.abs(sort_labels), 0)))
            res[i] = -sign
        loss.append(np.sum(np.where(res != np.sign(sort_labels), np.abs(sort_labels), 0)))
        idx = np.argmin(np.array(loss))
        if idx == len(sort_values):
            return sort_values[-1] + 1, loss[idx]
        return sort_values[idx], loss[idx]


    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        return misclassification_error(np.sign(y), self.predict(X))


