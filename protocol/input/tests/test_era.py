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
import matplotlib.pyplot as plt

from input import era, tests


def test_read_netcdf_file_erainterim():
    folder = 'era_interim'
    name = 'data.nc'

    data_path = os.path.join(tests.sample_data_path, 'era', folder, name)
    (msl_vec, units, coordinates) = era.read_netcdf_file(data_path)

    assert coordinates['lat'][0] == 36.625
    assert coordinates['lon'][0] == 353.625
    assert msl_vec.index[0].year == 1979

    folder = 'era_40'
    name = 'data.nc'

    data_path = os.path.join(tests.sample_data_path, 'era', folder, name)
    (msl_vec, units, coordinates) = era.read_netcdf_file(data_path)

    assert coordinates['lat'][0] == 36.625
    assert coordinates['lon'][0] == 353.625
    assert msl_vec.index[0].year == 1957


def test_combination_era_files():
    # Read era interim files
    folder = 'era_interim'
    name = 'data.nc'

    data_path = os.path.join(tests.sample_data_path, 'era', folder, name)
    (msl_vec_inter, units, coordinates) = era.read_netcdf_file(data_path)

    # Read era 40 files
    folder = 'era_40'
    name = 'data.nc'

    data_path = os.path.join(tests.sample_data_path, 'era', folder, name)
    (msl_vec_40, units, coordinates) = era.read_netcdf_file(data_path)

    # Combination of both files
    msl_vec = era.combination_era_files(msl_vec_inter, msl_vec_40)

    assert msl_vec.index[0].year == 1957
    assert msl_vec.index[-1].year == 2017


def test_combination_era_files_plot():
    # Read era interim files
    folder = 'era_interim'
    name = 'data.nc'

    data_path = os.path.join(tests.sample_data_path, 'era', folder, name)
    (msl_vec_inter, units, coordinates) = era.read_netcdf_file(data_path)

    # Read era 40 files
    folder = 'era_40'
    name = 'data.nc'

    data_path = os.path.join(tests.sample_data_path, 'era', folder, name)
    (msl_vec_40, units, coordinates) = era.read_netcdf_file(data_path)

    # Combination of both files
    msl_vec = era.combination_era_files(msl_vec_inter, msl_vec_40)

    # Plot graphic
    plt.figure()
    plt.plot(msl_vec_inter, 'g^')
    plt.plot(msl_vec_40, 'bs')
    plt.plot(msl_vec, 'r--')
    plt.legend(['Msl era interim', 'Msl era40', 'Msl combined'])
    plt.show()
