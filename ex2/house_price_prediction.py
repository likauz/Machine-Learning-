import math
import IMLearn.utils.utils
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    dataPrice = pd.read_csv(filename).dropna().drop_duplicates()
    dataPrice = dataPrice.drop(dataPrice[(dataPrice["id"] == 0)].index)
    floors = pd.get_dummies(dataPrice["floors"])
    dataPrice = dataPrice.drop(dataPrice[(dataPrice["price"] <= 0)].index)
    price = dataPrice['price']

    built_to_saleDay = dataPrice["yr_built"] - pd.to_numeric(dataPrice["date"].astype(str).apply(lambda x: x[:4]))
    built_to_renovation = dataPrice["yr_renovated"] - pd.to_numeric(
        dataPrice["date"].astype(str).apply(lambda x: x[:4]))

    features = ['bathrooms', 'bedrooms', 'sqft_living', 'view', 'grade', 'sqft_basement', 'sqft_above', 'yr_built',
                'waterfront', 'zipcode', 'condition', "sqft_lot", "sqft_living15", "sqft_lot15", "yr_renovated"]
    new_data = pd.concat([floors, dataPrice[features]], axis=1)
    for v in ["sqft_living", "sqft_lot", "sqft_above", "yr_built", "zipcode"]:
        new_data = new_data[new_data[v] > 0]
    for v in ["bathrooms", "sqft_basement", "grade", "waterfront", 'condition']:
        new_data = new_data[new_data[v] >= 0]
    new_data = new_data[new_data["waterfront"].isin([0, 1]) &
                        new_data["view"].isin(range(5)) &
                        new_data["condition"].isin(range(1, 6))]

    new_data["zipcode"] = (new_data["zipcode"] / 10).astype(int)
    new_data = pd.get_dummies(new_data, prefix='zipcode-', columns=['zipcode'])
    new_data.insert(0, "years_from_built_renovation",
                    pd.concat([built_to_saleDay, built_to_renovation], axis=1).max(axis=1))

    return new_data, price


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    var_y = np.std(y)
    for featureI in X:
        cov_xy = np.cov(X[featureI], y)[0][1]
        var_x = np.std(X[featureI])
        pearson_corelation = cov_xy / (var_x * var_y)
        pearson_corelation = np.round_(pearson_corelation, 4)
        fig = go.Figure([go.Scatter(y=y, x=X[featureI], mode='markers')],
                        layout=go.Layout(title=f"The Pearson Correlation is:{pearson_corelation}"
                                         , height=700, yaxis_title="Price", xaxis_title=featureI))
        fig.write_image(output_path + "pearson.correlation.%s.png" % featureI)


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    X, y = load_data("//Users//leeuziel//Desktop//house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    # feature_evaluation(X, y)

    # Question 3 - Split samples into training- and testing sets.
    train_X, train_y, test_X, test_y = split_train_test(X, y)
    # Question 4 - Fit model over increasing percentages of the overall training data
    mean_loss = []
    std_loss = []
    list_p = list(range(10, 101))
    for i in list_p:
        loss = []
        for j in range(10):
            estimator = LinearRegression(True)
            p_train_x = train_X.sample(frac=i / 100)
            p_train_y = train_y.reindex_like(p_train_x)
            estimator.fit(p_train_x.to_numpy(), p_train_y.to_numpy())
            loss.append(estimator.loss(test_X.to_numpy(), test_y.to_numpy()))
        std_loss.append(np.std(loss))
        mean_loss.append(np.mean(loss))
    arr = []
    std_loss = np.array(std_loss)
    mean_loss = np.array(mean_loss)

    for t in range(10):
        arr.append(go.Frame(data=[go.Scatter(x=list_p, y=mean_loss, mode="markers+lines", name="Means", marker=dict
        (color="black", opacity=.7))]))

    for p in range(len(arr)):
        y_minus = mean_loss - 2 * std_loss
        y_plus = mean_loss + 2 * std_loss

        arr[p]["data"] = (go.Scatter(x=list_p, y=y_minus, fill=None, mode="lines", line=dict(color="blue"),
                                     showlegend=False),
                          go.Scatter(x=list_p, y=y_plus, fill='tonexty', mode="lines",
                                     line=dict(color="blue"), showlegend=False),) + arr[p]["data"]
    fig = go.Figure(data=arr[0]["data"], frames=arr, layout=go.Layout(
        title="Average loss as function of training size with error ribbon ", xaxis=arr[0]["layout"]["xaxis"],
        yaxis=arr[0]["layout"]["yaxis"], xaxis_title="training size in percentage",
        yaxis_title="Mean loss"))
    fig.show()
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
