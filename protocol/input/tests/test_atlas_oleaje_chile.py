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

from input import atlas_oleaje_chile, tests


def test_read_time_series_file():
    name = 'wave.txt'
    data_path = os.path.join(tests.sample_data_path, 'atlas_oleaje_chile', name)
    data = atlas_oleaje_chile.read_time_series_file(data_path, path='.', null_values=(-99.9, -99.99, -9999, 990),
                                                    columns='std')

    assert data.iloc[40, :].loc['Hm0'] == 2.127836
    assert int(data.iloc[70, :].loc['DirM']) == 214


def test_read_time_series_file_plot():
    name = 'wave.txt'
    data_path = os.path.join(tests.sample_data_path, 'atlas_oleaje_chile', name)
    data = atlas_oleaje_chile.read_time_series_file(data_path, path='.', null_values=(-99.9, -99.99, -9999, 990),
                                                    columns='std')

    plt.figure()
    plt.plot(data['Hm0'], 'k')
    plt.show()
