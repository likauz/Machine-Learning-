import math
from math import atan2
from IMLearn.metrics import accuracy
import IMLearn
from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
import numpy as np
from typing import Tuple
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from utils import decision_surface, custom

pio.templates.default = "simple_white"


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * math.pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "//Users//leeuziel//IML.HUJI//datasets//linearly_separable.npy"),
                 ("Linearly Inseparable", "//Users//leeuziel//IML.HUJI//datasets//linearly_inseparable.npy")]:
        # Load dataset
        data_x, data_y = load_dataset(f)

        # Fit Perceptron and record loss in each fit iteration
        losses = []

        def help_func(cur, x, y):
            losses.append(cur._loss(data_x, data_y))
            return

        obj = Perceptron(callback=help_func)
        obj._fit(data_x, data_y)

        len_losses = list(range(1, len(losses) + 1))

        # Plot figure
        fig = go.Figure([go.Scatter(y=losses, x=len_losses, mode='markers + lines')],
                        layout=go.Layout(title=f"Loss performed in the x iteration when data is {n}",
                                         height=750, yaxis_title="Loss preformed in the x iteration",
                                         xaxis_title="Number of iterations"))
        fig.show()


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for i, f in enumerate(["//Users//leeuziel//IML.HUJI//datasets//gaussian1.npy",
                           "//Users//leeuziel//IML.HUJI//datasets//gaussian2.npy"]):
        # Load dataset
        X, y = load_dataset(f)

        # Fit models and predict over training set
        gnb_obj = GaussianNaiveBayes()
        lda_obj = LDA()
        gnb_obj.fit(X, y)
        lda_obj.fit(X, y)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        models_name = ["Gaussian Naive Bayes model", "Linear Discriminant Analysis model"]
        sample_symbol = np.array(['triangle-up', 'x', 'circle'])

        lims = np.array([X.min(axis=0), X.max(axis=0)]).T + np.array([-.4, .4])
        fig = make_subplots(rows=1, cols=2, subplot_titles=[rf"$\textbf{{{m}}}$" for m in models_name])
        for j, m in enumerate([gnb_obj, lda_obj]):
            accuracy_obj = round(accuracy(y, m.predict(X)), 3)
            fig.add_traces([decision_surface(m.predict, lims[0], lims[1], showscale=False),
                            go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers",
                                       marker=dict(symbol=sample_symbol[y.astype(int)], color=y,
                                                   colorscale=[custom[0], custom[-1], custom[2]]),
                                       text=f"accuracy: {accuracy_obj}", textposition="middle center"),
                            go.Scatter(x=m.mu_[:, 0], y=m.mu_[:, 1], mode='markers', marker=dict(color="white", symbol="circle")),
                            get_ellipse(m.mu_[0], m.cov_ if j == 1 else np.diag(m.vars_[0])),
                            get_ellipse(m.mu_[1], m.cov_ if j == 1 else np.diag(m.vars_[1])),
                            get_ellipse(m.mu_[2], m.cov_ if j == 1 else np.diag(m.vars_[2]))],
                           rows=1, cols=j+1)
            fig.layout.annotations[j].update(text=f"{models_name[j]} accuracy model: {accuracy_obj}")
        fig.update_layout(title=rf"$\textbf{{Gaussian Naive Bayes model and Linear Discriminant Analysis model"
                                "estimators with gaussian{i + 1}}}$",
                          margin=dict(t=100))
        fig.show()




if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
