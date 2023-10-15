from typing import NoReturn
from IMLearn.base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv

from IMLearn.metrics import misclassification_error


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """
    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        response_counter = {}  # response counter dictionary
        total_response = 0
        for res in y:
            if res in response_counter:
                response_counter[res] += 1
            else:
                response_counter[res] = 1
            total_response += 1

        self.classes_ = np.array(list(response_counter.keys()))

        self.pi_ = []
        response_index = {}
        index = 0
        for res_pro in self.classes_:
            response_index[res_pro] = index
            index += 1
            self.pi_.append(response_counter[res_pro]/total_response)
        self.pi_ = np.array(self.pi_)

        self.mu_ = []
        for res in self.classes_:
            self.mu_.append(1 / response_counter[res] * sum(X[i] for i in range(len(X)) if y[i] == res))
        self.mu_ = np.array(self.mu_)

        div = 1 / (len(X) - len(self.classes_))
        self.cov_ = div * sum((X[i] - self.mu_[response_index[y[i]]])[np.newaxis].transpose() @
                              (X[i] - self.mu_[response_index[y[i]]])[np.newaxis]for i in range(len(X)))

        self._cov_inv = np.array(inv(self.cov_))

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
        """
        y = []
        total_response = len(self.classes_)
        for cur_x in range(len(X)):
            a = self._cov_inv @ self.mu_[0][np.newaxis].transpose()
            b = np.log(self.pi_[0]) - 0.5 * self.mu_[0][np.newaxis] @ self._cov_inv @ self.mu_[0][np.newaxis] \
                .transpose()
            maximum = a.transpose() @ X[cur_x][np.newaxis].transpose() + b
            x = 0
            for k in range(total_response):
                a = self._cov_inv @ self.mu_[k][np.newaxis].transpose()
                b = np.log(self.pi_[k]) - 0.5 * self.mu_[k][np.newaxis] @ self._cov_inv @ self.mu_[k][np.newaxis]\
                    .transpose()
                cur = a.transpose() @ X[cur_x][np.newaxis].transpose() + b
                if cur > maximum:
                    maximum = cur
                    x = k
            y.append(self.classes_[x])
        return np.array(y)

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        total_response = len(self.classes_)
        total_likelihood = np.zeros((len(X), total_response))
        for cur_x in range(len(X)):
            for k in range(total_response):
                a = self._cov_inv @ self.mu_[k][np.newaxis].transpose()
                b = np.log(self.pi_[k]) - 0.5 * self.mu_[k][np.newaxis] @ self._cov_inv @ self.mu_[k][np.newaxis] \
                    .transpose()
                cur = a.transpose() @ X[cur_x][np.newaxis].transpose() + b
                total_likelihood[cur_x][k] = cur
        return total_likelihood

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
        return misclassification_error(y, self.predict(X))

if __name__ == '__main__':
    X, y = np.array([[1, 1], [0.8, 1.1], [3, 3], [3.5, 3.2]]), np.array([1, 1, 0, 0])
    lda = LDA()
    lda.fit(X, y)
    print(lda.predict(np.array([[0.4, 0.9], [2.5,2.8 ]])))
    print(lda.likelihood(np.array([[0.4, 0.9], [2.5,2.8 ]])))
    print(lda.loss(X, y))
