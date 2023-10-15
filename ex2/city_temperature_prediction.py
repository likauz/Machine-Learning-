
import plotly.graph_objects as go
import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    data = pd.read_csv(filename).dropna()
    data.drop(data[(data["Country"] == 0)].index)
    data.drop(data[(data["City"] == 0)].index)
    data.drop(data[(data["Date"] == 0)].index)
    data.drop(data[(data["Temp"] == 0)].index)
    data = data[data["Temp"] > -70]
    data.insert(0, "DayOfYear", data.apply(lambda x: __get_num_of_day(x.Day, x.Month), axis=1))
    temp = data["Temp"]

    return data

def __get_num_of_day(day, month)-> int:
    month_day = sum([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][:month])
    return month_day + day


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    X = load_data("//Users//leeuziel//Desktop//City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    israel = X[X["Country"] == "Israel"]
    # 1)
    israel.insert(0, "YearToString", israel["Year"].astype(str))
    fig = px.scatter(israel, x="DayOfYear", y="Temp", color="YearToString"
                     , title="Temperature as function of the day in year").show()
    israel.drop(["YearToString"], axis=1)
    # 2)
    std_month = israel.groupby(["Month"])["Temp"].std()
    px.bar(std_month, color_discrete_sequence=["pink"]
           , title="The standard deviation for each month of the daily temperatures").show()

    # Question 3 - Exploring differences between countries
    group = X.groupby(['Country', 'Month'])['Temp'].agg(['mean', 'std']).reset_index()
    px.line(group, x='Month', y='mean', line_group='Country', color='Country', error_y='std').show()

    # Question 4 - Fitting model for different values of `k`
    train_X, train_y, test_X, test_y = split_train_test(israel.loc[:, 'DayOfYear'], israel.loc[:, 'Temp'], 0.75)
    loss = []

    for k in range(1, 11):
        polynom_obj = PolynomialFitting(k)
        polynom_obj.fit(train_X.to_numpy(), train_y.to_numpy())
        obj_loss = polynom_obj.loss(test_X.to_numpy(), test_y.to_numpy())
        loss.append(round(obj_loss, 2))
        print(f"The test error for polynomial model of {k} degree: " + str(obj_loss))
    fig = go.Figure([go.Bar(x=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], y=loss, text=loss)],
                        layout=go.Layout(title=r"$\text{ The test error for polynomial model of K degree}$",
                         height=800,  xaxis_title="Polynomial degree", yaxis_title="Test error"))
    fig.show()

    # Question 5 - Evaluating fitted model on different countries
    polynom_obj6 = PolynomialFitting(5)
    polynom_obj6.fit(israel["DayOfYear"], israel["Temp"])
    The_Netherlands = X[X["Country"] == "The Netherlands"]
    South_Africa = X[X["Country"] == "South Africa"]
    Jordan = X[X["Country"] == "Jordan"]

    country_loss = []
    country_loss.append(round(polynom_obj6.loss(The_Netherlands["DayOfYear"], The_Netherlands["Temp"]), 2))
    country_loss.append(round(polynom_obj6.loss(South_Africa["DayOfYear"], South_Africa["Temp"]), 2))
    country_loss.append(round(polynom_obj6.loss(Jordan["DayOfYear"], Jordan["Temp"]),2))
    fig = go.Figure([go.Bar(x=["The Netherlands", "South Africa", "Jordan"], y=country_loss, text=country_loss)],
     layout=go.Layout(title=r"$\text{ The test error for each Country}$",
     height=800, xaxis_title="Country", yaxis_title="Test error"))
    fig.show()
