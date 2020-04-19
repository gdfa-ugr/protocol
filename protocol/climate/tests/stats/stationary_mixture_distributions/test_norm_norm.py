#!/usr/bin/env python
# coding: utf-8
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

try:
    # noinspection PyUnresolvedReferences, PyCompatibility
    from builtins import *  # noqa
except ImportError:
    pass

# import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from climate.stats.stationary_mixture_distributions import norm_norm
# from climate.tests import util

# # Build an absolute path from current script
# current_path = os.path.abspath(os.path.dirname(__file__))
# # Build relative paths from the former script path
# sample_data_path = os.path.join(current_path, '..', '..', 'data')


def test_pdf():
    # data_simar = util.read_sample_simar(data_path=sample_data_path)
    # data_column = 'Hm0'

    data = np.concatenate((np.random.normal(1, 2, (250, 1)), np.random.normal(7.85, 1.2, (500, 1))))
    np.random.shuffle(data)
    data = pd.Series(data[:, 0])

    initial_params = norm_norm.initial_params(data)
    pdf = norm_norm.pdf(data, initial_params)

    plt.plot(pdf.index, pdf, color='orange')

    plt.show()


def test_cdf():
    # data_simar = util.read_sample_simar(data_path=sample_data_path)
    # data_column = 'Hm0'

    data = np.concatenate((np.random.normal(1, 2, (250, 1)), np.random.normal(7.85, 1.2, (500, 1))))
    np.random.shuffle(data)
    data = pd.Series(data[:, 0])

    initial_params = norm_norm.initial_params(data)
    cdf = norm_norm.cdf(data, initial_params)

    plt.plot(cdf.index, cdf, color='orange')

    plt.show()
