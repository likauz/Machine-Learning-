import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

from sklearn.metrics import roc_curve, auc

from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.metrics import misclassification_error, mean_square_error
from IMLearn.utils import split_train_test

from IMLearn.model_selection import cross_validate
import plotly.graph_objects as go

from utils import custom



def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """

    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines",
                                 marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    values, weights = [], []

    def callback(**kwargs):
        values.append(kwargs["val"])
        weights.append(kwargs["weights"])

    return callback, values, weights

def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    l1_losses = np.inf
    l2_losses = np.inf
    for eta in etas:
        l1 = L1(weights=init.copy())
        l2 = L2(weights=init.copy())
        callback, vals, weights = get_gd_state_recorder_callback()
        gradient_descent = GradientDescent(learning_rate=FixedLR(eta), callback=callback, out_type="best")

        # L1
        gradient_descent.fit(l1, X=None, y=None)
        plot_descent_path(L1, np.array(weights), title=f"L1, eta={eta}").show()
        convergence_fig = go.Figure()
        vals_list = [i for i in range(0 ,len(vals))]
        convergence_fig.add_trace(go.Scatter(x=vals_list, y=vals, mode='lines+markers', name="Graph-L1-Norm"))
        convergence_fig.update_layout(title=f"Convergence Rate for L1 and L2 norm with eta = {eta}")
        convergence_fig.update_yaxes(title="$norm$")
        convergence_fig.update_xaxes(title="$Number of Iterations$")
        l1_cur_loss = l1.compute_output()
        if l1_cur_loss < l1_losses: l1_losses = l1_cur_loss
        # L2
        callback, vals, weights = get_gd_state_recorder_callback()
        gradient_descent = GradientDescent(learning_rate=FixedLR(eta), callback=callback)
        gradient_descent.fit(l2, X=None, y=None)
        plot_descent_path(L2, np.array(weights), title=f"L2, eta={eta}").show()

        numbers_of_iter = [i for i in range(0 ,len(vals))]
        convergence_fig.add_trace(go.Scatter(x=numbers_of_iter, y=vals, name="Graph-L2-Norm")).show()
        l2_cur_loss = l2.compute_output()
        if l2_cur_loss < l2_losses: l2_losses = l2_cur_loss

    print("losses l1: " + str(l1_losses))
    print("losses l2: " + str(l2_losses))


def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    # Optimize the L1 objective using different decay-rate values of the
    # exponentially decaying learning rate
    # Plot algorithm's convergence for the different values of gamma
    y_values, weights_ = [], []
    max = 0
    for gamma in gammas:
        callback, values, weights = get_gd_state_recorder_callback()
        GradientDescent(learning_rate=ExponentialLR(eta, gamma), callback=callback).fit(L1(init), X=None, y=None)
        if len(values) > max:
            max = len(values)
        if gamma == 0.95:
            weights_ = weights
        y_values.append(values)
    fig = go.Figure()
    iterations = list(range(1, max + 1))
    fig.add_traces([go.Scatter(x=iterations, y=y_values[0], name="decay_rate: 0.9", marker=dict(color='blue')),
                    go.Scatter(x=iterations, y=y_values[1], name="decay_rate: 0.95", marker=dict(color='green')),
                    go.Scatter(x=iterations, y=y_values[2], name="decay_rate: 0.99", marker=dict(color="orange")),
                    go.Scatter(x=iterations, y=y_values[3], name="decay_rate: 1", marker=dict(color="red"))])
    fig.update_layout(title="convergence rate for all decay rates", xaxis_title="number of iterations",
                      yaxis_title="loss value").show()
    # Plot descent path for gamma=0.95
    plot_descent_path(L1, np.array(weights_), f"for L1 norm with decay-rate: 0.95").show()


def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()


    # Plotting convergence rate of logistic regression over SA heart disease data
    logistic_model = LogisticRegression()
    logistic_model.fit(X_train.to_numpy(), y_train.values)
    c = [custom[0], custom[-1]]
    lr_proba = logistic_model.predict_proba(X_train.to_numpy())
    fpr, tpr, thresholds = roc_curve(y_train, lr_proba)
    fig = go.Figure(data=[go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="pink", dash='dash'),
                         name="Random Class"),
              go.Scatter(x=fpr, y=tpr, mode='markers+lines', text=thresholds, showlegend=False, marker_size=5,
                         marker_color=c[1][1],
                         hovertemplate="<b>Threshold:</b>%{text:.3f}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}")],
        layout=go.Layout(title=rf"$\text{{ROC Curve Of Fitted Model - the auc}}={auc(fpr, tpr):.6f}$",
            xaxis=dict(title=r"$\text{False Positive Rate }$"),
            yaxis=dict(title=r"$\text{True Positive Rate }$")))
    fig.show()
    print("the optimal alpha is:" + str(thresholds[np.argmax(tpr - fpr)]))
    logistic_model.alpha_ = thresholds[np.argmax(tpr - fpr)]
    print("the test error is: " + str(logistic_model.loss(X_test.to_numpy(), y_test.values)))

    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter
    lam_val = np.linspace(0.01, 2, 15)
    # L1
    train_scores_l1, validation_scores_l1 = [], []
    for lam in lam_val:
        logistic_model = LogisticRegression(penalty="l1", lam=lam)
        train_score, validation_score = cross_validate(logistic_model, X_train.to_numpy(), y_train.values.flatten(),
                                                       mean_square_error, 5)
        validation_scores_l1.append(validation_score)
        train_scores_l1.append(train_score)

    best_lam_index = int(np.argmin(np.array(validation_scores_l1)))
    logistic_l1 = LogisticRegression(penalty="l1", lam=lam_val[best_lam_index])
    logistic_l1.fit(X_train.to_numpy(), y_train.values)
    test_error_l1 = logistic_l1.loss(X_test.to_numpy(), y_test.values)
    best_lam_val = lam_val[best_lam_index]
    print("Best lambda for L1:" + str(best_lam_val))
    print("Model test error for L1:" + str(test_error_l1))

    # L2
    train_scores_l2, validation_scores_l2 = [], []
    for lam in lam_val:
        logistic_model = LogisticRegression(penalty="l2", lam=lam)
        train_score, validation_score = cross_validate.cross_validate(logistic_model,X_train.to_numpy(),
                                                                      y_train.values.flatten(),
                                                                      mean_square_error, 5)
        validation_scores_l2.append(validation_score)
        train_scores_l2.append(train_score)
    best_lam_index = int(np.argmin(np.array(validation_scores_l2)))
    logistic_l2 = LogisticRegression(penalty="l2", lam=lam_val[best_lam_index])
    logistic_l2.fit(X_train.to_numpy(), y_train.values)
    test_error_l2 = logistic_l2.loss(X_test.to_numpy(), y_test.values)
    best_lam_val = lam_val[best_lam_index]
    print("Best lambda for L2:" + str(best_lam_val))
    print("Model test error for L2:" + str(test_error_l2))


if __name__ == '__main__':
    np.random.seed(0)
    # compare_fixed_learning_rates()
    # compare_exponential_decay_rates()
    fit_logistic_regression()
