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
from scipy.stats.mstats import mode
from matplotlib import pyplot as plt
from climate.util import plot


def find_timestep(data, n_random_values=10):
    # Random position
    pos_ini = np.random.randint(0, data.shape[0], n_random_values, 'int64')
    pos_end = pos_ini + 1

    time = data.index.values
    time_ini = time[pos_ini]
    time_end = time[pos_end]

    time_difference = time_end - time_ini
    time_step = mode(time_difference)[0][0]

    return pd.to_timedelta(time_step)


def erase_null_values(data, method='all'):

    # Delete null values
    data_clean = data.dropna(axis=0, how=method)

    return data_clean


def erase_duplicated_time_indices(data, method='first'):
    # Delete duplicated indices
    data_clean = data[~data.index.duplicated(keep=method)]

    return  data_clean


def find_missing_values(data, t_step):
    """

    Args:
        data ():
        t_step ():

    Returns:

    """
    # Erase of the rows with any null values
    data_clean = erase_null_values(data, method='any')

    # Erase duplicate date
    data_unique = erase_duplicated_time_indices(data_clean, method='first')

    # Calculation of the difference between timesteps
    time_difference = data_unique.index.to_series().diff()

    # Find the values where the difference is different from the t_step
    val = np.where(time_difference.values != np.timedelta64(t_step))
    val = val[0][1:]

    # Gap duration and final position
    gap_duration = time_difference.iloc[val]

    pos_end = pd.Series(gap_duration.index.values, name='pos_fin')
    duration = pd.Series(gap_duration.values, name='duration')

    # Initial position of the gaps
    previous_val = val - 1
    pos_ini = time_difference.iloc[previous_val].index
    pos_ini = pd.Series(pos_ini, name='pos_ini')

    # Dataframe con las posiciones de los huecos
    df = pd.concat([pos_ini, pos_end, duration], axis=1)

    return df


def fill_missing_values(data, time_step, technique='interpolation', method='nearest', limit=24, limit_direction='both'):
    data_fill = -1

    # Reindex to the actual timestep to change gaps for nan
    new_index = pd.date_range(data.index[0], data.index[-1], freq=time_step)
    data_reindex = data.reindex(new_index)

    # Interpolate nan values
    if technique == 'interpolation':
        data_fill = data_reindex.interpolate(method=method, axis=0, limit=limit, limit_direction=limit_direction)

# TODO Add analogos

    return data_fill


def missing_values_report(data, data_gaps):
    # Total gap duration
    total_gap_dur = data_gaps.loc[:, 'duration'].sum()
    # Convert to hours
    total_gap_hours = total_gap_dur.total_seconds() / 3600
    # Difference in hours between data ini and data end
    diff = (data.index[-1] - data.index[0]).total_seconds() / 3600
    # Ratio
    gaps_ratio = (total_gap_hours / diff) * 100
    # Describe
    gap_summary = (data_gaps.loc[:, 'duration']).describe()
    # Append
    gap_summary['ratio'] = gaps_ratio

    return gap_summary


def plot_missing_values(data, data_gaps, data_column='', title='', var_name='', var_unit='', fig_filename='',
                        circular=False, label='observations'):
    # Get style
    plot.get_default_plot_style()
    # Get values
    values = plot.get_values(data, data_column)
    # Get var label
    var_label = plot.get_var_label(var_name, var_unit)
    # Plot
    ax = plt.axes()
    ax.plot(values.index, values, ':', alpha=0.3, label=label)
    ax.plot(values.loc[data_gaps.loc[:, 'pos_ini']], '.k', markersize=10, label='Gaps_initial_postion')
    ax.plot(values.loc[data_gaps.loc[:, 'pos_fin']], '.r', markersize=10, label='Gaps_final_postion')
    ax.legend(loc='upper right', facecolor='white')
    plot.plot_title(ax, title)

    if circular is True:
        circular_labels, circular_ticks = plot.get_circular_ticks()
        plt.yticks(circular_ticks, circular_labels)

    ax.set_ylabel(var_label)
    # fixing the x tick labels are all squashed together
    plt.gcf().autofmt_xdate()

    plot.save_figure(fig_filename)
