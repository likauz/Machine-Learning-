import numpy as np
from IMLearn.base import BaseEstimator
from typing import Callable, NoReturn

from IMLearn.learners.classifiers import DecisionStump
from IMLearn.metrics import misclassification_error


class AdaBoost(BaseEstimator):
    """
    AdaBoost class for boosting a specified weak learner

    Attributes
    ----------
    self.wl_: Callable[[], BaseEstimator]
        Callable for obtaining an instance of type BaseEstimator

    self.iterations_: int
        Number of boosting iterations to perform

    self.models_: List[BaseEstimator]
        List of fitted estimators, fitted along the boosting iterations
    """

    def __init__(self, wl: Callable[[], BaseEstimator], iterations: int):
        """
        Instantiate an AdaBoost class over the specified base estimator

        Parameters
        ----------
        wl: Callable[[], BaseEstimator]
            Callable for obtaining an instance of type BaseEstimator

        iterations: int
            Number of boosting iterations to perform
        """
        super().__init__()
        self.wl_ = wl
        self.iterations_ = iterations
        self.models_, self.weights_, self.D_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit an AdaBoost classifier over given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.weights_ = []
        self.models_ = []
        self.D_ = np.ones(len(X)) / len(X)
        for t in range(self.iterations_):
            h_t = self.wl_()
            h_t.fit(X, y * self.D_)
            h_t_predict = h_t.predict(X)
            self.models_.append(h_t)
            e_t = np.sum(np.where(h_t_predict != y, self.D_, 0))
            w_t = 0.5 * np.log((1-e_t)/e_t)
            self.weights_.append(w_t)
            self.D_ *= np.exp((-1) * w_t * y * h_t_predict)
            self.D_ /= np.sum(self.D_)
        self.models_ = np.array(self.models_)
        self.weights_ = np.array(self.weights_)


    def _predict(self, X):
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
        """
        return np.sign(np.sum(self.weights_[i] * self.models_[i].predict(X) for i in range(self.iterations_)))

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

    def partial_predict(self, X:np.ndarray, T: int) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimators
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        T: int
            The number of classifiers (from 1,...,T) to be used for prediction

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        res = np.zeros(len(X))
        for i in range(T):
            res += self.models_[i].predict(X) * self.weights_[i]
        return np.sign(res)



    def partial_loss(self, X: np.ndarray, y: np.ndarray, T: int) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        T: int
            The number of classifiers (from 1,...,T) to be used for prediction

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        return misclassification_error(np.sign(y), self.partial_predict(X, T))


