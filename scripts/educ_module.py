import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
from scipy.optimize import minimize
import ipywidgets as widgets
from IPython.display import display
from matplotlib.ticker import FuncFormatter
import warnings
warnings.filterwarnings('ignore')


def knn(neighbors=1):

    model = KNeighborsRegressor(neighbors)

    X = np.array([[-1, -1.5], [-2, -1.5], [-3, -2], [1, 1], [2, 1], [3, 3]])
    random_y_values = np.array([2, 3, 4, 5, 4, 1])
    to_predict = [0, 0]

    model.fit(X, random_y_values)
    dist, ind = model.kneighbors([to_predict])

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 7))

    circle = plt.Circle(to_predict, max(dist[0]), color='g', alpha=.2)
    axes[0].add_artist(circle)

    axes[0].plot(to_predict[0], to_predict[1], 'x', color='g', mew=3)

    axes[0].scatter(X[:, 0], X[:, 1], color='black')

    closest_points = X[ind[0]]

    axes[0].set_title('Distances')
    x_coords = closest_points.transpose()[0]
    y_coords = closest_points.transpose()[1]
    axes[0].scatter(x_coords, y_coords, color='r')

    for i in range(len(random_y_values)):
        position = X[i]
        axes[0].text(position[0] - .05, position[1] + .07,
                     str(random_y_values[i]))

    num_points = len(ind[0])
    axes[1].set_xlim([0, 7])
    axes[1].set_ylim([0, 6])

    values = []
    for i in range(num_points):
        value = random_y_values[ind[0][i]]
        axes[1].vlines(x=i + 1, ymin=0, ymax=value,
                       color='r', linewidths=15)
        values.append(value)
    axes[1].hlines(y=np.mean(values), xmin=0, xmax=12,
                   linestyles='dashed', linewidths=2, color='g')

    axes[1].set_title('Values of k closest')
    axes[1].set_xlabel('k')
    axes[1].set_ylabel('Value')
    plt.show()
    print('Predicted Value: ', np.mean(values))


def regressiont(included_points=16, dependent="Wolf Population"):

    data = pd.read_csv(
        'data/wolf_and_elk_in_yellowstone.csv',
        thousands=',').drop(
        'Notes',
        axis=1).dropna().reset_index(
            drop=True)

    ignore = 16 - included_points  # indexes to ignore

    # prediction values
    yp = data[dependent][ignore:]
    tp = data['Year'][ignore:]

    # values
    y = data[dependent]
    t = data['Year']

    # loss function
    def res(pars):
        a, b = pars
        yp_hat = a + b * tp
        res = yp - yp_hat
        return sum(res**2)

    # optimal paramters
    a, b = minimize(res, (-6.06621468e+04, 3.04230758e+01)).x

    # parameters for constant
    if included_points == 1:
        a, b = 480, 0

    y_hat = a + b * t  # precited values
    fitline = [y_hat.iloc[0], y_hat.iloc[-1]]  # fit line end points
    fittime = [t.iloc[0], t.iloc[-1]]  # fit line time values

    y_prime = a + b * 2013  # predicted point

    ax1 = plt.scatter(t, y, c='b')  # points
    ax2 = plt.plot(fittime, fitline, 'r--')  # fitline
    ax3 = plt.plot(tp, yp, 'rs')  # points used to predict
    ax4 = plt.scatter(t.iloc[-1] + 1, y_prime, c='g',
                      marker='^')  # predicted point

    # graph labels
    plt.xticks(t.append(pd.Series([2013]))[::2])
    plt.xlabel('Year')
    plt.ylabel(dependent)

    # display graph and predicted value
    plt.show()
    print('Predicted 2013 ' + dependent + ':', round(y_prime))
