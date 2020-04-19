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

from climate.stats import fitting


def simple_stationary(data, var_name, fitting_function, fitting_params=None, cumulative=False):
    x, y, params = fitting.fit(data, var_name, fitting_function, fitting_params, cumulative=cumulative)

    return x, y, params


def extremal_annual_maxima(data, var_name, fitting_params=None, cumulative=False):
    fitting_function = st.genextreme

    x, y, params = fitting.fit(data, var_name, fitting_function, fitting_params, cumulative=cumulative)

    # Return period
    return_period = 1/(1-y)

    return x, y, return_period, params


def extremal_peaks_over_thresholds():
    print('Hola')

    return
