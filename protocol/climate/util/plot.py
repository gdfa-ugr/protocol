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
from scipy import stats as st
from matplotlib import pyplot as plt


def plot_title(ax, title):
    ax.set_title(title, y=1.08)


def get_circular_ticks():
    circular_labels = ('N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW',
                       'NNW', 'N')
    circular_ticks = (0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5, 180, 202.5, 225, 247.5, 270, 292.5, 315, 337.5, 360)

    return circular_labels, circular_ticks


def get_months_ticks():
    months_labels = ('Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'July', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec')
    months_ticks = range(0, 12)

    return months_labels, months_ticks


def get_x_ticks(x, lim, step=5):
    x_labels = range(x.min(), x.max()+1, step)
    x_ticks = range(lim[0], lim[1]+1, step)

    return list(x_labels), list(x_ticks)


def save_figure(fig_filename):
    if fig_filename != '':
        plt.savefig(fig_filename, transparent=False)
        plt.close()
    else:
        plt.show()


def get_values(data, data_column):
    if data_column != '':
        values = data[data_column]
    else:
        values = data

    return values


def get_var_label(var_name, var_unit):
    label = ''
    if var_name != '' and var_unit != '':
        label = '{} ({})'.format(var_name, var_unit)

    return label


def get_default_plot_style(style='darkgrid', axes_style=None, context='notebook'):
    if axes_style is None:
        axes_style = {}
    axes_style.update({'legend.frameon': True})

    sns.set_context(context)
    sns.set_style(style, axes_style)


def calculation_width(dist):
    # remove rows with same index
    dist = dist[~dist.index.duplicated(keep='first')]

    x_min = dist.index.min()
    x_max = dist.index.max()
    width = (x_max - x_min) / (len(dist.index) - 1)

    return width, x_min, x_max


def change_bar_width(ax, new_value):
    for patch in ax.patches:
        current_width = patch.get_width()
        diff = current_width - new_value

        # we change the bar width
        patch.set_width(new_value)

        # we recenter the bar
        patch.set_x(patch.get_x() + diff * .5)


def calculation_x_limits(data, cut=3, clip=(-np.inf, np.inf)):
    bw = st.gaussian_kde(data).scotts_factor() * data.std(ddof=1)

    support_min = max(data.min() - bw * cut, clip[0])
    support_max = min(data.max() + bw * cut, clip[1])

    return support_min, support_max
