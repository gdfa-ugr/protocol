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
import seaborn as sns
from matplotlib import pyplot as plt

from climate.util import plot


def fit(data, data_column, fit_function, fitting_params=None, cumulative=False, x_min=None, x_max=None, x_step=200):
    if fitting_params is None:
        fitting_params = dict()

    if x_min is None or x_max is None:
        x_min, x_max = plot.calculation_x_limits(data[data_column])

    x = np.linspace(x_min, x_max, x_step)

    # fitting_params is a keyworded variable length argument
    params = fit_function.fit(data[data_column], **fitting_params)

    # params is variable length argument
    if cumulative:
        y = fit_function.cdf(x, *params)
    else:
        y = fit_function.pdf(x, *params)

    return x, y, params


def plot_fit_kde(x, y, data, data_name, title='', var_name='', var_unit='', kde=True, hist=False, cumulative=False,
             fig_filename='', fit_label='fitting', hist_label='empirical', kde_label='kernel'):
    plot.get_default_plot_style()

    # kw_params is a keyworded variable length argument
    kw_params = {'hist_kws': {'label': hist_label, 'edgecolor': 'black'}, 'kde_kws': {'label': kde_label}}
    if cumulative:
        # noinspection PyTypeChecker
        kw_params['hist_kws'].update({'cumulative': True})
        # noinspection PyTypeChecker
        kw_params['kde_kws'].update({'cumulative': True})

    var_label = plot.get_var_label(var_name, var_unit)

    ax = sns.distplot(data[data_name], kde=kde, hist=hist, **kw_params)

    ax.plot(x, y, label=fit_label)

    ax.legend(loc='upper right', facecolor='white')

    plot.plot_title(ax, title)

    ax.set_xlabel(var_label)

    plot.save_figure(fig_filename)


def plot_fit_kde_ecdf(x_gev, y_gev, x_ecdf, y_ecdf, x_ecdf_kde, y_ecdf_kde, var_name='', var_unit='', label_fit='',
                      label_observation='', label_kde='', title='', fig_filename=''):

    plot.get_default_plot_style()

    var_label = plot.get_var_label(var_name, var_unit)

    ax = plt.axes()
    ax.plot(x_gev, y_gev, 'k', label=label_fit)
    ax.plot(x_ecdf, y_ecdf, '.b', label=label_observation)
    ax.plot(x_ecdf_kde, y_ecdf_kde, 'r', label=label_kde)
    ax.legend(loc='upper right', facecolor='white')
    plot.plot_title(ax, title)

    ax.set_xlabel(var_label)
    # fixing the x tick labels are all squashed together
    plt.gcf().autofmt_xdate()

    plot.save_figure(fig_filename)



