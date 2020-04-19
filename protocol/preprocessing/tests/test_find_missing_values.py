#!/usr/bin/env python
# coding: utf-8
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

try:
    # noinspection PyUnresolvedReferences, PyCompatibility
    from builtins import *  # noqa
except ImportError:
    pass

import os
import pytest
import pandas as pd
import matplotlib.pyplot as plt


from climate import read
from preprocessing import missing_values, tests


def test_erase_null_values():
    # Define null values
    method = 'any'

    # Read simar file
    data_path = os.path.join(tests.sample_data_path, 'simar')
    # noinspection PyTypeChecker
    data_simar, _ = read.simar('SIMAR_1052046_short_gap', data_path)

    # Erase null values
    data = missing_values.erase_null_values(data_simar, method)

    assert '1958-01-04 08' not in data.index
    assert '1958-01-04 19' not in data.index
    assert '1960-12-31 18' not in data.index


def test_find_timestep():
    # Read simar file
    data_path = os.path.join(tests.sample_data_path, 'simar')
    # noinspection PyTypeChecker
    data_simar, _ = read.simar('SIMAR_1052046_short', data_path)

    t_step = missing_values.find_timestep(data_simar)

    assert t_step == pd.timedelta(hours=1)


def test_find_missing_values():
    # Read simar file
    data_path = os.path.join(tests.sample_data_path, 'simar')
    # noinspection PyTypeChecker
    data_simar, _ = read.simar('SIMAR_1052046_short_gap', data_path)

    # Calculate the time step
    time_step = missing_values.find_timestep(data_simar)

    data_gaps = missing_values.find_missing_values(data_simar, time_step)

    # Representation of the gaps
    fig = plt.figure()
    ax = fig.gca()

    ax.plot(data_simar.loc[:, 'Hm0'])
    ax.plot(data_simar.loc[data_gaps.loc[:, 'pos_ini'], 'Hm0'], 'k.', markersize=10)
    ax.plot(data_simar.loc[data_gaps.loc[:, 'pos_fin'], 'Hm0'], 'k.', markersize=10)

    fig.show()


def test_fill_missing_values():
    # Read simar file
    data_path = os.path.join(tests.sample_data_path, 'simar')
    # noinspection PyTypeChecker
    data_simar, _ = read.simar('SIMAR_1052046_short_gap', data_path)

    # Calculate the time step
    time_step = missing_values.find_timestep(data_simar)

    # Fill missing values
    data_fill = missing_values.fill_missing_values(data_simar, time_step, technique='interpolation', method='nearest',
                                                   limit=24, limit_direction='both')

    tolerance = 0.01
    assert data_fill.loc['1958-01-04 08', 'Hm0'] == pytest.approx(2.1, tolerance)
    assert data_fill.loc['1958-01-04 12', 'Tp'] == pytest.approx(10.5, tolerance)
    assert data_fill.loc['1958-01-04 14', 'Tp'] == pytest.approx(10.6, tolerance)
    assert data_fill.loc['1960-12-31 19', 'Hm0'] == pytest.approx(0.6, tolerance)


def test_missing_values_report():
    # Input
    file_name = 'gaps_report.csv'
    path = os.path.join('.', '..', '..', 'report', 'tests', 'output', 'tables')
    # Read simar file
    data_path = os.path.join(tests.sample_data_path, 'simar')
    # noinspection PyTypeChecker
    data_simar, _ = read.simar('SIMAR_1052046_short_gap', data_path)

    # Calculate the time step
    time_step = missing_values.find_timestep(data_simar)
    # Find gaps
    data_gaps = missing_values.find_missing_values(data_simar, time_step)
    # Gaps report
    data_gaps_report = missing_values.missing_values_report(data_simar, data_gaps)
    missing_values.missing_values_report_to_file(data_gaps_report, file_name, path)
    # Plot
    missing_values.plot_missing_values(data=data_simar, data_column='Hm0', data_gaps=data_gaps,
                                       title='', var_name='Hm0', var_unit='m', fig_filename='', circular=False,
                                       label='Hm0')
