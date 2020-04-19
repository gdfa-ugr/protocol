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

from input import ww3, tests


def test_read_time_series_file():
    name = 'Pto_1.txt'
    data_path = os.path.join(tests.full_data_path, 'ww3', 'Ptos', name)
    data = ww3.read_time_series_file(data_path, path='.', null_values=(-99.9, -99.99, -9999, 990),
                                     columns='all')

    assert data.loc['2005-02-07 21', 'Hs'] == 1.45
    assert data.loc['2016-06-10 00', 'DirM'] == 113.84


def test_read_time_series_file_plot():
    name = 'Pto_1.txt'
    data_path = os.path.join(tests.full_data_path, 'ww3', 'Ptos', name)
    data = ww3.read_time_series_file(data_path, path='.', null_values=(-99.9, -99.99, -9999, 990),
                                     columns='std')
    t = data.index

    plt.figure()
    f, axarr = plt.subplots(5, sharex=str('all'))
    axarr[0].plot(t, data['Hs'])
    axarr[0].set_ylabel('Hs (m)')
    axarr[1].plot(t, data['Tp'])
    axarr[1].set_ylabel('Tp (s)')
    axarr[2].plot(t, data['DirM'])
    axarr[2].set_ylabel('D_w (o)')
    axarr[3].plot(t, data['VelV'])
    axarr[3].set_ylabel('W_v (m/s)')
    axarr[4].plot(t, data['DirV'])
    axarr[4].set_ylabel('D_v (o)')
    plt.show()
