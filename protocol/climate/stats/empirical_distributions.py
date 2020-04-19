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
import pandas as pd
import seaborn as sns
import scipy.stats as st
from matplotlib import pyplot as plt

from climate.util import plot


def epdf_histogram(data, bins=0):
    data_size = data.shape[0]

    if bins == 0:
        q1, q3 = st.scoreatpercentile(data, [25, 75])
        iqr = q3 - q1

        h = 2 * iqr / (data_size ** (1 / 3))
        if h == 0:
            bins = np.ceil(np.sqrt(data.size))
        else:
            bins = np.ceil((data.max() - data.min()) / h)
    elif bins == -1:
        bins = np.ceil(1 + np.log(data_size) / np.log(2))
    elif bins == -2:
        bins = np.ceil(np.sqrt(data_size))

    if np.unique(data).shape[0] == 2:
        bins = 2

    if np.unique(data).shape[0] == 1:
        dist = None
        xc = None
    else:
        [n, xc] = np.histogram(data, int(bins))
        xc = xc[:-1] + np.diff(xc) / 2  # convert edge positions to center positions

        dx = xc[1] - xc[0]
        dist = n / (dx * data_size)

    return pd.Series(dist, index=xc)


def ecdf_histogram(data):
    data = data.sort_values()

    dist = np.arange(1, len(data)+1) / len(data)

    return pd.Series(dist, index=data)


def ecdf_sm(data):
    from statsmodels.distributions.empirical_distribution import ECDF

    empirical_cumulative_dist_function = ECDF(data)

    return pd.Series(empirical_cumulative_dist_function.y, index=empirical_cumulative_dist_function.x)


def plot_ecdf_sm_dots(data, title='', var_name='', var_unit='', fig_filename='', data_column='',
                      label='Observations'):
    plot.get_default_plot_style()

    values = plot.get_values(data, data_column)
    var_label = plot.get_var_label(var_name, var_unit)

    ax = plt.axes()
    ax.plot(values.index, values, '.', label=label)
    ax.legend(loc='upper right', facecolor='white')
    plot.plot_title(ax, title)

    ax.set_xlabel(var_label)
    # fixing the x tick labels are all squashed together
    plt.gcf().autofmt_xdate()

    plt.show()
    plot.save_figure(fig_filename)


def plot_empirical(dist, fig_filename=''):
    # width is in axis units
    width, _, _ = plot.calculation_width(dist)

    plt.bar(dist.index, dist, width, edgecolor='black', linewidth=0.5)

    plot.save_figure(fig_filename)


def plot_empirical_fit(dist, fitting_function, params, x_step=200, cumulative=False, fig_filename=''):
    # width is in axis units
    width, x_min, x_max = plot.calculation_width(dist)

    plt.bar(dist.index, dist, width, edgecolor='black', linewidth=0.5)

    x = np.linspace(x_min, x_max, x_step)

    if cumulative:
        cdf = getattr(fitting_function, 'cdf')
        p = cdf(x, *params)
    else:
        pdf = getattr(fitting_function, 'pdf')
        p = pdf(x, *params)

    plt.plot(x, p, color='orange')

    plot.save_figure(fig_filename)


def kde_scipy(data, bw='scott', gridsize=100, cut=3, clip=(-np.inf, np.inf)):
    # NOTE: Cumulative distributions are currently only implemented in statsmodels

    # noinspection PyTypeChecker
    kde = st.gaussian_kde(data, bw_method=bw)

    x_min, x_max = plot.calculation_x_limits(data, cut, clip)
    grid = np.linspace(x_min, x_max, gridsize)

    # noinspection PyArgumentList
    y = kde(grid)

    return pd.Series(y, index=grid)


def kde_sm(data, kernel='gau', bw='scott', gridsize=None, cut=3, clip=(-np.inf, np.inf), cumulative=False):
    import statsmodels.nonparametric.api as smnp

    fft = kernel == 'gau'
    kde = smnp.KDEUnivariate(data)
    # noinspection PyTypeChecker
    kde.fit(kernel, bw, fft, gridsize=gridsize, cut=cut, clip=clip)
    if cumulative:
        grid, y = kde.support, kde.cdf
    else:
        grid, y = kde.support, kde.density

    return pd.Series(y, index=grid)


def plot_kde(data_empirical, data_kernel, cumulative, title='', var_name='', var_unit='', fig_filename='',
             circular=False, label_empirical='', label_kernel=''):

    plot.get_default_plot_style()

    values_empirical = plot.get_values(data_empirical, '')
    var_label = plot.get_var_label(var_name, var_unit)

    ax = plt.axes()
    ax.plot(values_empirical.index, values_empirical, label=label_empirical)
    if cumulative:
        values_kernel = plot.get_values(data_kernel, '')
        ax.plot(values_kernel.index, values_kernel, label=label_kernel)
    ax.legend(loc='upper right', facecolor='white')

    plot.plot_title(ax, title)

    if circular:
        circular_labels, circular_ticks = plot.get_circular_ticks()
        plt.yticks(circular_ticks, circular_labels)

    plot.plot_title(ax, title)
    ax.set_xlabel(var_label)

    if cumulative:
        ax.set_ylabel('CDF')
    else:
        ax.set_ylabel('PDF')

    # fixing the x tick labels are all squashed together
    plt.gcf().autofmt_xdate()

    plot.save_figure(fig_filename)
