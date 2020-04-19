#!/usr/bin/env python
# coding: utf-8
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

try:
    # noinspection PyUnresolvedReferences, PyCompatibility
    from builtins import *  # noqa
except ImportError:
    pass

import numpy as np
import os
from scipy import interpolate
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import genpareto
from scipy.stats import genextreme
from scipy.stats import exponweib, lognorm, norm, weibull_min
from climate.util import time_series
from climate.stats.empirical_distributions import ecdf_histogram


def fit_distribution(data, fit_type, x_min, x_max, n_points=1000):
    # Initialization of the variables
    param, x, cdf, pdf = [-1, -1, -1, -1]

    if fit_type == 'exponweib':
        x = np.linspace(x_min, x_max, n_points)

        # Fit data to the theoretical distribution
        param = exponweib.fit(data, 1, 1, scale=02, loc=0)
        # param = exponweib.fit(data, fa=1, floc=0)
        # param = exponweib.fit(data)

        cdf = exponweib.cdf(x, param[0], param[1], param[2], param[3])
        pdf = exponweib.pdf(x, param[0], param[1], param[2], param[3])

    elif fit_type == 'lognorm':
        x = np.linspace(x_min, x_max, n_points)

        # Fit data to the theoretical distribution
        param = lognorm.fit(data, loc=0)

        cdf = lognorm.cdf(x, param[0], param[1], param[2])
        pdf = lognorm.pdf(x, param[0], param[1], param[2])

    elif fit_type == 'norm':
        x = np.linspace(x_min, x_max, n_points)

        # Fit data to the theoretical distribution
        param = norm.fit(data, loc=0)

        cdf = norm.cdf(x, param[0], param[1])
        pdf = norm.pdf(x, param[0], param[1])

    elif fit_type == 'weibull_min':
        x = np.linspace(x_min, x_max, n_points)

        # Fit data to the theoretical distribution
        param = weibull_min.fit(data, floc=0)
        
        cdf = weibull_min.cdf(x, param[0], param[1], param[2])
        pdf = weibull_min.pdf(x, param[0], param[1], param[2])

    return param, x, cdf, pdf


def remove_zeros(data):
    data.replace(to_replace=0, value=0.01, inplace=True)

    return data


def pp_plot(x_cdf, cdf, ecdf):
    # First it is necessary to interpolate the values of the theoretical cdf at the points of the empirical cdf
    xppplot = ecdf.index
    yppplot_emp = ecdf

    f = interpolate.interp1d(x_cdf, cdf)

    yppplot_teo = f(xppplot)  # use interpolation function returned by `interp1d`

    return yppplot_emp, yppplot_teo


def qq_plot(x_cdf, cdf, ecdf):
    # First and last value of the distributions are set to zero and one.
    cdf[0] = 0
    cdf[-1] = 1
    ecdf[0] = 0
    ecdf[-1] = 1

    # First it is necessary to interpolate the values of the theoretical cdf at the points of the empirical cdf
    xqqplot = ecdf
    yqqplot_emp = ecdf.index
    f = interpolate.interp1d(cdf, x_cdf)

    yqqplot_teo = f(xqqplot)  # use interpolation function returned by `interp1d`

    return yqqplot_emp, yqqplot_teo


def plot_goodness_of_fit(cdf, ecdf, modf, var_name, x_cdf, yppplot_emp, yqqplot_emp, yppplot_teo, yqqplot_teo):
    # Plot goodness of fit
    figure, axes = plt.subplots(2, 2, figsize=(20, 20))
    axes[0, 0].plot(ecdf.index, ecdf, '.')
    axes[0, 0].plot(x_cdf, cdf)
    axes[0, 0].set_ylabel('CDF')
    axes[0, 0].set_title('Cumulative Distribution Function')
    axes[0, 0].legend(['Empirical', 'Theoretical'])
    axes[0, 0].grid()

    axes[0, 1].semilogy(ecdf.index, ecdf, '.')
    axes[0, 1].semilogy(x_cdf, cdf)
    axes[0, 1].set_ylabel('CDF')
    axes[0, 1].set_title('Cumulative Distribution Function (Log scale)')
    axes[0, 1].legend(['Empirical', 'Theoretical'])
    axes[0, 1].grid()

    axes[1, 0].plot(yppplot_emp, yppplot_teo)
    axes[1, 0].plot([0, 1], [0, 1])
    axes[1, 0].set_xlabel('F_{emp}')
    axes[1, 0].set_ylabel('F_{teo}')
    axes[1, 0].set_title('PP - Plot')
    axes[1, 0].grid()

    axes[1, 1].plot(yqqplot_emp, yqqplot_teo)
    axes[1, 1].plot([np.min(modf.loc[:, var_name]), 1.3*np.max(modf.loc[:, var_name])], [np.min(modf.loc[:, var_name]),
                                                                                       1.3*np.max(modf.loc[:, var_name])])
    axes[1, 1].set_xlabel('F_{emp}')
    axes[1, 1].set_ylabel('F_{teo}')
    axes[1, 1].set_title('QQ - Plot')
    axes[1, 1].set_xlim([np.min(modf.loc[:, var_name]), 1.3*np.max(modf.loc[:, var_name])])
    axes[1, 1].set_ylim([np.min(modf.loc[:, var_name]), 1.3*np.max(modf.loc[:, var_name])])
    axes[1, 1].grid()

    figure.savefig(os.path.join('output', 'analisis', 'graficas', 'goodness_of_fit_' + var_name + '.png'))


def plot_goodness_of_fit_nest(cdf, ecdf, modf, var_name, x_cdf, yppplot_emp, yqqplot_emp, yppplot_teo, yqqplot_teo):
    # Plot goodness of fit
    figure, axes = plt.subplots(2, 2, figsize=(20, 20))
    axes[0, 0].plot(ecdf.index, ecdf, '.')
    axes[0, 0].plot(x_cdf, cdf)
    axes[0, 0].set_ylabel('CDF')
    axes[0, 0].set_title('Cumulative Distribution Function')
    axes[0, 0].legend(['Empirical', 'Theoretical'])
    axes[0, 0].grid()

    axes[0, 1].semilogy(ecdf.index, ecdf, '.')
    axes[0, 1].semilogy(x_cdf, cdf)
    axes[0, 1].set_ylabel('CDF')
    axes[0, 1].set_title('Cumulative Distribution Function (Log scale)')
    axes[0, 1].legend(['Empirical', 'Theoretical'])
    axes[0, 1].grid()

    axes[1, 0].plot(yppplot_emp, yppplot_teo)
    axes[1, 0].plot([0, 1], [0, 1])
    axes[1, 0].set_xlabel('F_{emp}')
    axes[1, 0].set_ylabel('F_{teo}')
    axes[1, 0].set_title('PP - Plot')
    axes[1, 0].grid()

    axes[1, 1].plot(yqqplot_emp, yqqplot_teo)
    axes[1, 1].plot([np.min(modf.loc[:, var_name]), 1.3*np.max(modf.loc[:, var_name])], [np.min(modf.loc[:, var_name]),
                                                                                       1.3*np.max(modf.loc[:, var_name])])
    axes[1, 1].set_xlabel('F_{emp}')
    axes[1, 1].set_ylabel('F_{teo}')
    axes[1, 1].set_title('QQ - Plot')
    axes[1, 1].set_xlim([np.min(modf.loc[:, var_name]), 1.3*np.max(modf.loc[:, var_name])])
    axes[1, 1].set_ylim([np.min(modf.loc[:, var_name]), 1.3*np.max(modf.loc[:, var_name])])
    axes[1, 1].grid()

    figure.savefig(os.path.join('output', 'analisis', 'graficas', 'goodness_of_fit_nest_' + var_name + '.png'))

