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

from input import snirh, tests


def test_read_file():
    file_name = 'river_flow.csv'
    path = os.path.join(tests.sample_data_path, 'snirh')
    data = snirh.read_file(file_name, path)

    assert True


def test_read_file_plot():
    file_name = 'river_flow.csv'
    path = os.path.join(tests.sample_data_path, 'snirh')
    data = snirh.read_file(file_name, path)

    plt.figure()
    plt.plot(data, 'k')
    plt.show()
