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

from input import tidal_model_driver, tests


def test_read_time_series_file():
    name = 'time_series.out'
    data_path = os.path.join(tests.sample_data_path, 'tidal_model_driver', name)
    data = tidal_model_driver.read_time_series_file(data_path)

    assert data.iloc[0, 0] == 0.4209
    assert data.iloc[149, 0] == 0.7242


def test_read_time_series_file_plot():
    name = 'time_series.out'
    data_path = os.path.join(tests.sample_data_path, 'tidal_model_driver', name)
    data = tidal_model_driver.read_time_series_file(data_path)

    plt.figure()
    plt.plot(data, 'k')
    plt.legend(['Astronomical tide (m)'])
    plt.show()
    plt.interactive(False)
