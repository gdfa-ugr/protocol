#!/usr/bin/env python
# coding: utf-8
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

try:
    # noinspection PyUnresolvedReferences, PyCompatibility
    from builtins import *  # noqa
except ImportError:
    pass

from future.utils import bytes_to_native_str as n  # to specify str type on both Python 2/3

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.interpolate import UnivariateSpline

from climate.third_party.windrose import plot_windrose
from climate.util import plot, time_series


def get_summary(data):
    summary = data.describe()

    return summary


def plot_series(data, title='', var_name='', var_unit='', fig_filename='', data_column='', circular=False,
                show_trends=False, label='observations'):
    plot.get_default_plot_style()

    values = plot.get_values(data, data_column)
    var_label = plot.get_var_label(var_name, var_unit)

    ax = plt.axes()
    if show_trends is False:
        ax.plot(values.index, values, linewidth=0.5)
    else:
        # TODO Comprobar que funciona correctamente con más ejemplos

        # UnivariateSpline needs conversion from DateTimeIndex to float
        splined_index = values.index.values.astype('d')
        splined_values = UnivariateSpline(splined_index, values)

        ax.plot(values.index, values, ':', alpha=0.3, label=label)
        ax.plot(values.index, splined_values(splined_index))

        ax.legend(loc='upper right', facecolor='white')

    plot.plot_title(ax, title)

    if circular:
        circular_labels, circular_ticks = plot.get_circular_ticks()
        plt.yticks(circular_ticks, circular_labels)

    ax.set_ylabel(var_label)
    # fixing the x tick labels are all squashed together
    plt.gcf().autofmt_xdate()

    plot.save_figure(fig_filename)


def plot_rose(data, data_column, dir_column, title='', var_name='', var_unit='', fig_filename='',
              legend_position='upper left', xlabels=('E', 'N-E', 'N', 'N-W', 'W', 'S-W', 'S', 'S-E')):
    plot.get_default_plot_style(context='talk')

    var_label = plot.get_var_label(var_name, var_unit)

    ax = plot_windrose(data, kind='bar', var_name=data_column, direction_name=dir_column, normed=True, opening=0.8,
                       edgecolor=n(b'white'))

    plot.plot_title(ax, title)

    ax.set_xlabel(var_label)
    ax.legend(loc=legend_position, fontsize='small')
    ax.set_xticklabels(xlabels)

    plot.save_figure(fig_filename)


def plot_histogram(data, title='', var_name='', var_unit='', fig_filename='', rug=False, data_column='',
                   circular=False, kernel=False, bins='auto'):
    plot.get_default_plot_style()

    values = plot.get_values(data, data_column)
    var_label = plot.get_var_label(var_name, var_unit)

    # Remove null values
    values_clean = values.dropna(axis=0, how='all')
    ax = sns.distplot(values_clean, rug=rug, hist_kws={'edgecolor': 'black'}, kde=kernel, bins=bins,
                      norm_hist=True)

    plot.plot_title(ax, title)

    if circular:
        circular_labels, circular_ticks = plot.get_circular_ticks()
        plt.xticks(circular_ticks, circular_labels)

    ax.set_xlabel(var_label)

    plot.save_figure(fig_filename)


def plot_scatter(data, x_column, y_column, title='', x_var_name='', x_var_unit='',
                 y_var_name='', y_var_unit='', circular=None, fig_filename='', mark_size=15):
    plot.get_default_plot_style()

    x_var_label = plot.get_var_label(x_var_name, x_var_unit)
    y_var_label = plot.get_var_label(y_var_name, y_var_unit)

    ax = sns.jointplot(x_column, y_column, data=data, stat_func=None, joint_kws={'s': mark_size},
                       marginal_kws={'hist_kws': {'edgecolor': 'black'}})

    plt.subplots_adjust(top=0.9)
    plt.gcf().suptitle(title)

    if circular == 'x':
        circular_labels, circular_ticks = plot.get_circular_ticks()
        ax.ax_joint.set_xticks(circular_ticks)
        ax.ax_joint.set_xticklabels(circular_labels, rotation=45)
        plt.subplots_adjust(bottom=0.15)
    elif circular == 'y':
        circular_labels, circular_ticks = plot.get_circular_ticks()
        ax.ax_joint.set_yticks(circular_ticks)
        ax.ax_joint.set_yticklabels(circular_labels)
        plt.subplots_adjust(left=0.15)

    ax.set_axis_labels(x_var_label, y_var_label)

    plot.save_figure(fig_filename)


def plot_variability(data, frecuency, title='', var_name='', var_unit='', x_step=1, fliersize=1, linewidth=0.5,
                     color='skyblue', palette=None, fig_filename='', circular=False):
    plot.get_default_plot_style()

    var_label = plot.get_var_label(var_name, var_unit)

    # gettattr return the value of the named attribute,
    # i.e. data.index.daily is equivalent to getattr(data.index, 'daily')
    x = getattr(data.index, frecuency)
    ax = sns.boxplot(x=x, y=data, fliersize=fliersize, linewidth=linewidth, color=color,
                     palette=palette)

    plot.plot_title(ax, title)

    if circular:
        circular_labels, circular_ticks = plot.get_circular_ticks()
        plt.yticks(circular_ticks, circular_labels)

    if frecuency == 'month':
        months_labels, months_ticks = plot.get_months_ticks()
        plt.xticks(months_ticks, months_labels)
    else:
        if x_step > 1:
            x_labels, x_ticks = plot.get_x_ticks(x, ax.get_xlim(), x_step)
            plt.xticks(x_ticks, x_labels)
        # fixing the x tick labels are all squashed together
        ax.xaxis_date()
        plt.gcf().autofmt_xdate()

    ax.xaxis.grid(True)
    ax.set_xlabel('')
    ax.set_ylabel(var_label)

    plot.save_figure(fig_filename)


# TODO Comprobar que está funcionando correctamente
def plot_anual_variability(data, daily_agg='mean', plot_agg=np.mean, ci='sd', color='darkblue', fig_filename=''):
    plot.get_default_plot_style()

    # Aggregate by day
    data_aggregated = getattr(data.resample('D'), daily_agg)
    # Split years
    timeseries_data = time_series.generate_year_stack(data_aggregated())
    # Fill gaps (one week maximum)
    data_filled = timeseries_data.fillna(method='ffill', axis=1, limit=7)
    # Remove incomplete years
    data_clean = data_filled.dropna()
    # Convert to numpy array
    data_values = data_clean.values

    sns.tsplot(data=data_values, estimator=plot_agg, ci=ci, color=color)

    plot.save_figure(fig_filename)
