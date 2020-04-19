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

from input.util import dates


def read_time_series_file(data_path, path='.', null_values=(-99.9, -99.99, -999, -9999, 990), columns='std'):
    df = pd.read_table(os.path.join(path, data_path), skiprows=1, delim_whitespace=True, header=None,
                       na_values=null_values)
    time_col = df.iloc[:, 2:6]
    time_col.columns = list(range(4))
    date_col = dates.extract_date(time_col)
    df.index = date_col
    df.drop(df.columns[2:6], axis=1, inplace=True)
    df.columns = ['lon', 'lat', 'Hs', 'Tp', 'DirM', 'u', 'v', 'VelV', 'DirV']

    if columns == 'std':
        df = df.drop(['lon', 'lat', 'u', 'v'], 1)

    return df
