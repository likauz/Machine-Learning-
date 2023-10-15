from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import math

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    # raise NotImplementedError()
    samples = np.random.normal(10, 1, 1000)
    obj = UnivariateGaussian()
    obj.fit(samples)
    var = obj.var_
    exp = obj.mu_
    print(exp, var)

    # Question 2 - Empirically showing sample mean is consistent
    distances = []
    for i in range(10, 1001, 10):
        obj.fit(samples[:i])
        distances.append(abs(obj.mu_ - 10))
    go.Figure(
        [go.Scatter(x=[i for i in range(10, 1001, 10)], y=distances, mode='markers+lines', name=r'$\widehat\mu$')],
        layout=go.Layout(title=r"$\text{ Estimation of Expectation As Function Of Number Of Samples}$",
                         height=800, xaxis_title="Number of samples", yaxis_title="Expections diffreces")).show()


    # Question 3 - Plotting Empirical PDF of fitted model
    go.Figure([go.Scatter(x=samples, y=obj.pdf(samples), mode='markers', line=dict(width=4, color="rgb(204,68,83)"),
                          name=r'$\widehat\mu$')], layout=go.Layout(title=r"$\text{ PDF Samples$",
                                                                    height=800, xaxis_title="samples",
                                                                    yaxis_title="PDF")).show()

def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mean = np.array([0, 0, 4, 0]).transpose()
    var = np.array([[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]])

    samples = np.random.multivariate_normal(mean, var, 1000)
    obj = MultivariateGaussian()
    obj.fit(samples)
    var = obj.cov_
    exp = obj.mu_
    print(exp)
    print(var)

    # Question 5 - Likelihood evaluation
    x = np.linspace(-10, 10, 200)
    y = np.linspace(-10, 10, 200)
    z = np.zeros((200, 200))
    for i in range(200):
        for j in range(200):
            mu = np.asarray([x[i], 0, y[j], 0])
            z[j][i] = MultivariateGaussian.log_likelihood(mu, var, samples)

    go.Figure(go.Heatmap(x=x, y=y, z=z, colorscale='blues'), layout=go.Layout(title="The log-likelihood",
                                                                              height=600, width=800, xaxis_title="f1",
                                                                              yaxis_title="f3")).show()
    # Question 6 - Maximum likelihood
    maximum_index = np.argmax(z)
    print(x[maximum_index//200], y[maximum_index % 200])

if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
