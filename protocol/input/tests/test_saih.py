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

from input import saih, tests


def test_read_file():
    name = 'river_flow.txt'
    data_path = os.path.join(tests.sample_data_path, 'saih', name)
    data = saih.read_file(data_path)

    assert data.shape[0] == 22274
    assert int(data.iloc[3905, 0]) == 12
    assert int(data.iloc[17405, 0]) == 25


def test_read_file_plot():
    name = 'river_flow.txt'
    data_path = os.path.join(tests.sample_data_path, 'saih', name)
    data = saih.read_file(data_path)

    plt.figure()
    plt.plot(data, 'k')
    plt.show()
