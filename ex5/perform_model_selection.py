from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso, ridge_regression

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots




def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """

    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions

    data = pd.DataFrame(np.random.uniform(-1.2, 2, n_samples))
    data["y"] = data.apply(lambda x: (x+3) * (x+2) * (x+1) * (x-1) * (x-2), axis=1)
    data['noise'] = pd.Series(np.random.normal(0, noise, n_samples))
    data['noise_y'] = data.apply(lambda d: d.y + d.noise, axis=1)
    data.columns = ['x', 'y', 'noise', 'noise_y']
    slice = math.ceil(data.shape[0] * (2/3))
    data['len'] = range(len(data))
    data['train_or_test'] = data.apply(lambda d: d.len <= slice, axis=1)
    fig_no_noise = px.scatter(data, x='x', y='y', render_mode='markers+lines').update_traces(marker=dict(color='wheat'))
    fig_noise = px.scatter(data, x='x', y='noise_y', color='train_or_test')
    fig_total = go.Figure(data=fig_noise.data + fig_no_noise.data)
    fig_total.show()
    train_x, train_y, test_x, test_y = split_train_test(data['x'], data['noise_y'], 0.66)

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    validation_loss_arr = []
    train_loss_arr = []
    for k in range(11):
        poly_fitting = PolynomialFitting(k)
        train_loss, validation_loss = cross_validate(poly_fitting, train_x, train_y, mean_square_error)
        train_loss_arr.append(train_loss)
        validation_loss_arr.append(validation_loss)
    go.Figure([go.Scatter(y=train_loss_arr, x=list(range(11)), mode='markers + lines', name="$Train_loss$"),
               go.Scatter(y=validation_loss_arr, x=list(range(11)), mode='markers + lines', name="$Validation_loss$")],
              layout=go.Layout(title=r"$\text{Train losses}$",
                               height=650, yaxis_title="The Loss on the train",
                               xaxis_title="Polynomial degree")).show()


# Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    k_best = np.argmin(np.array(validation_loss_arr))
    poly_fitting = PolynomialFitting(k_best)
    poly_fitting.fit(train_x, train_y)
    print("Test error on polynomial of best degree", k_best,
          round(poly_fitting.loss(test_x, test_y), 2))



def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    X, y = datasets.load_diabetes(return_X_y=True)
    train_x, train_y, test_x, test_y = X[:n_samples], y[:n_samples], X[n_samples:], y[n_samples:]

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    lamdas = np.linspace(0.0001, 1.1, n_evaluations)
    train_lasso_arr, valid_lasso_arr, train_ridge_arr, valid_ridge_arr = [], [], [], []
    for lamda in lamdas:
        ridge_model = RidgeRegression(lamda)
        lasso_model = Lasso(alpha=lamda)
        train_lasso, valid_lasso = cross_validate(lasso_model, train_x, train_y,
                                                  mean_square_error)
        train_ridge, valid_ridge = cross_validate(ridge_model, train_x, train_y,
                                                  mean_square_error)
        train_lasso_arr.append(train_lasso)
        valid_lasso_arr.append(valid_lasso)
        train_ridge_arr.append(train_ridge)
        valid_ridge_arr.append(valid_ridge)
    go.Figure([go.Scatter(y=train_ridge_arr, x=lamdas, mode='markers + lines', name="$Train loss$"),
                   go.Scatter(y=valid_ridge_arr, x=lamdas, mode='markers + lines', name="$Validation loss$")],
                  layout=go.Layout(title=r"$\text{Ridge Regression}$",
                                   height=650, yaxis_title="Loss on train / validation",
                                   xaxis_title="lambda ")).show()
    go.Figure([go.Scatter(y=train_lasso_arr, x=lamdas, mode='markers + lines', name="$Train loss$"),
                   go.Scatter(y=valid_lasso_arr, x=lamdas, mode='markers + lines', name="$Validation loss$")],
                  layout=go.Layout(title=r"$\text{Lasso Regression}$",
                                   height=650, yaxis_title="Loss on train / validation",
                                   xaxis_title="lambda ")).show()


    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    best_lambda_ridge = lamdas[np.argmin(np.array(valid_ridge_arr))]
    best_lambda_lasso = lamdas[np.argmin(np.array(valid_lasso_arr))]

    ridge_model = RidgeRegression(best_lambda_ridge)
    lasso_model = Lasso(alpha=best_lambda_lasso)
    regression_model = LinearRegression()

    ridge_model.fit(train_x, train_y)
    lasso_model.fit(train_x, train_y)
    regression_model.fit(train_x, train_y)

    print(f"Ridge Regression model loss- test data : {ridge_model.loss(test_x, test_y)} with lambda = {best_lambda_ridge}")
    print(f"Lasso Regression model loss- test data : {mean_square_error(lasso_model.predict(test_x), test_y)}"
          f" with lambda = {best_lambda_lasso}")
    print(f"Linear Regression model loss- test data :", regression_model.loss(test_x, test_y))

if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    select_polynomial_degree(noise=0)
    select_polynomial_degree(n_samples=1500, noise=10)
    select_regularization_parameter()

