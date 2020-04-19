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


def generate_year_stack(data):
    # Locate changes of year
    positions_diff = np.diff(data.index.year)
    split_positions = np.where(positions_diff == 1)[0] + 1

    data_splitted = np.split(data.values, split_positions)
    data_stacked = pd.DataFrame(data_splitted)  # Pandas stack (allows different shapes, filling with NaN)

    return data_stacked


def values_over_threshold(data, threshold):
    """ Extract the threshold crossing values for characterization of the storm and calm cycles

    Args:
        data (pd.Series): Time series data
        threshold (int): Threshold that separates storm (cycle) and calm periods

    Returns:
        np.array: Indices with the position of the start of the storm cycles
        np.array: Indices with the position of the end of the storm cycles
    """
    # Find cycles starting and ending positions
    # noinspection PyTypeChecker
    cycles_data = data.where(data >= threshold).to_sparse().sp_index

    cycles_start = cycles_data.blocs
    cycles_duration = cycles_data.blengths
    cycles_end = cycles_start + cycles_duration

    return cycles_start, cycles_end


def interpolation_series(data, index_start, index_end):
    start = data[index_start]
    end = data[index_end]

    group = start.index.asi8

    interpolation_start = start.to_frame('value')
    interpolation_start['group'] = group

    interpolation_end = end.to_frame('value')
    interpolation_end['group'] = group

    interpolation = pd.concat([interpolation_start, interpolation_end])

    return interpolation


def interpolation_nearest(group, threshold):
    return group.iloc[(group['value']-threshold).abs().argsort()[:1]]


def interpolation_boundaries_index(data, cycles_start, cycles_end, interpolation_freq, interpolation_method, threshold):
    if cycles_start[0] == 0:
        interpolation_start = interpolation_series(data, cycles_start[1:]-1, cycles_start[1:])
    else:
        interpolation_start = interpolation_series(data, cycles_start-1, cycles_start)

    interpolation_end = interpolation_series(data, cycles_end-1, cycles_end)

    interpolation_df = pd.concat([interpolation_start, interpolation_end])
    interpolation_df.index.name = 'time'

    interpolation_groups = interpolation_df.groupby('group').resample(interpolation_freq).asfreq()
    interpolation = interpolation_groups['value'].interpolate(interpolation_method)

    new_cycles_limits_indexes = interpolation.reset_index('time').groupby(level='group', group_keys=False).apply(
        interpolation_nearest, threshold)['time'].values

    new_cycles_limits_indexes = np.unique(new_cycles_limits_indexes)

    return new_cycles_limits_indexes


def cross_time_threshold(cross_time, data, cycles_start, cycles_end, threshold):
    """ Function for making the duration of the cycle more accurate. This function takes the introduced cross time and
    distributes it at the beginning and the end of the cycle to improve the duration between the threshold crossing and
    the start of the cycle.

    Args:
        cross_time (pd.Timedelta): Time between the instant when the thresolhold is reached and the instant
            when the storm starts. Normally this value is set equal to the frequency of the data
        data (pd.Series): Time series data
        cycles_start (np.array):
        cycles_end (np.array): Indices with the position of the start of the storm cycles
        threshold (): Indices with the position of the end of the storm cycles

    Returns:
        np.array: Indices with the new and improved position of the start of the storm cycles
        np.array: Indices with the new and improved position of the end of the storm cycles
    """
    new_cycles_start_index = data[cycles_start].index
    new_cycles_end_index = data[cycles_end-1].index

    # Avoid adding values in the cycle beginning or ending if it already has the threshold value
    new_cycles_start_index = data[new_cycles_start_index].where(
        data[new_cycles_start_index] != threshold).dropna().index
    new_cycles_end_index = data[new_cycles_end_index].where(
        data[new_cycles_end_index] != threshold).dropna().index

    new_cycles_start = pd.Series(threshold, index=new_cycles_start_index - cross_time/2)
    new_cycles_end = pd.Series(threshold, index=new_cycles_end_index + cross_time/2)

    # Remove values out of data range
    new_cycles_start = new_cycles_start[new_cycles_start.index > data.index[0]]
    new_cycles_end = new_cycles_end[new_cycles_end.index < data.index[-1]]

    return new_cycles_start, new_cycles_end


def extreme_indexes(data, extreme_start, extreme_end):
    indexes = np.arange(len(data))

    condition = (indexes >= extreme_start[:, np.newaxis]) & (indexes < extreme_end[:, np.newaxis])
    extreme = np.any(condition, axis=0)

    return extreme


def near_events(data, cycles_start, cycles_end, minimum_interarrival_time, interpolation):
    if interpolation:
        near_cycles = data.index[cycles_start[1:]] - data.index[cycles_end[:-1] - 1] < minimum_interarrival_time
    else:
        near_cycles = data.index[cycles_start[1:] - 1] - data.index[cycles_end[:-1]] < minimum_interarrival_time

    return near_cycles
