import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from IMLearn.metrics import accuracy
from IMLearn.utils import utils
import utils
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case

    train = []
    test = []
    adaboost = AdaBoost(DecisionStump, n_learners)
    adaboost.fit(train_X, train_y)
    for t in range(1, n_learners):
        train.append(adaboost.partial_loss(train_X, train_y, t))
        test.append(adaboost.partial_loss(test_X, test_y, t))
    go.Figure([
        go.Scatter(x=list(range(1, n_learners)), y=train, mode='markers + lines', name='Train Error'),
        go.Scatter(x=list(range(1, n_learners)), y=test, mode='markers + lines', name='Test Error')]) \
        .update_layout(title="AdaBoost error", xaxis=dict(title="number of models")).show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    fig = make_subplots(rows=2, cols=2, subplot_titles=[rf"$\textbf{{{m}}}$" for m in T],
                        horizontal_spacing=0.01, vertical_spacing=.03)
    for i, t in enumerate(T):
        fig.add_traces([decision_surface(lambda x: adaboost.partial_predict(x, t), lims[0], lims[1], showscale=False),
                        go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                                   marker=dict(color=test_y))], rows=int(i / 2) + 1, cols=(i % 2) + 1)

    fig.update_layout(title=rf"$\textbf{{Ada-booster algorithm for different number of iterations}}$",
                      margin=dict(t=100)) \
        .update_xaxes(visible=False).update_yaxes(visible=False)
    fig.show()

    # find accuracy
    accuracie_array = np.array([accuracy(test_y, adaboost.partial_predict(test_X, T=t)) for t in T])
    best_acc = np.argmax(accuracie_array)
    best_T = T[best_acc]
    best_accuracy = str(accuracie_array[best_acc])

 
    # Question 3: Decision surface of best performing ensemble

    go.Figure([utils.decision_surface(lambda X: adaboost.partial_predict(X, best_T), lims[0], lims[1]),
               go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                          marker=dict(color=test_y, size=8, line=dict(color="black", width=0.2)))])\
        .update_layout(xaxis_range=[-1, 1],yaxis_range=[-1, 1], title=f" The Best Performing Ensemble For Adaboost for {best_T} number of iterations, \n"
              f"accuracy  {best_accuracy}").show()


    # Question 4: Decision surface with weighted samples
    normalize = 5 * adaboost.D_ / np.max(adaboost.D_)
    d_normalize = np.array(normalize)
    colors_dict = np.array(['pink', 'orange'])
    train_y = np.where(train_y > 0, 1, 0).astype(int)

    go.Figure([utils.decision_surface(lambda X: adaboost.partial_predict(X, best_T), lims[0], lims[1], showscale=False),
               go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers", showlegend=False,
                          marker=dict(color=train_y, size=d_normalize * 3,
                                      line=dict(color=colors_dict[train_y], width=0.2)))]).update_layout(
        xaxis_range=[-1, 1], yaxis_range=[-1, 1],
        title=(f"The Best Performing Ensemble With Weighted Samples for {best_T} number of iterations "
              f"accuracy = {best_accuracy}")).show()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)
    fit_and_evaluate_adaboost(0.4)
