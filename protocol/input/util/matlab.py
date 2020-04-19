#!/usr/bin/env python
# coding: utf-8
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

try:
    # noinspection PyUnresolvedReferences, PyCompatibility
    from builtins import *  # noqa
except ImportError:
    pass

import pandas as pd
import os
from scipy.io import loadmat
from input.util.dates import convert_datenum_datetime


def read_time_series_matlab(data_path, variables, path='.', time_column_pos=None, time_var=None):
    mat_data = loadmat(os.path.join(path, data_path), squeeze_me=True)

    time_series = []

    single_variable = False
    if not isinstance(variables, (list, tuple)):
        single_variable = True
        variables = [variables]

    for variable in variables:
        data = mat_data[variable]

        if data.ndim > 1:
            python_data = pd.DataFrame(data)
        else:
            python_data = pd.Series(data)

        matlab_t = None
        if time_column_pos is not None:
            matlab_t = data[:, time_column_pos]
        elif time_var is not None:
            matlab_t = mat_data[time_var]

        if matlab_t is not None:
            python_dt = convert_datenum_datetime(matlab_t)
            python_data.index = python_dt

        time_series.append(python_data)

    if single_variable:
        time_series = time_series[0]

    return time_series
