#!/usr/bin/env python
# coding: utf-8
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

try:
    # noinspection PyUnresolvedReferences, PyCompatibility
    from builtins import *  # noqa
except ImportError:
    pass

import scipy.stats as st

from climate import analysis
from climate import tests
from climate.stats import fitting


def test_simple_stationary():
    data_simar = tests.read_sample_simar()
    data_column = 'Hm0'
    cumulative = False

    x, y, _ = analysis.simple_stationary(data_simar, data_column, st.weibull_min, cumulative=cumulative)
    fitting.plot_fit_kde(x, y, data_simar, data_column, title='Weibull fitting', var_name='$H_s$', var_unit='m',
                         kde=True, hist=True, cumulative=cumulative)
