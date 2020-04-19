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


def read_time_series_file(data_path, path='.', null_values=(-99.9, -99.99, -9999, 990), columns='std'):
    df = pd.read_table(os.path.join(path, data_path), header=None, delim_whitespace=True, skiprows=1,
                       na_values=null_values)
    date_col = dates.extract_date(df.iloc[:, 0:4])
    df.index = date_col
    df.drop(df.columns[0:4], axis=1, inplace=True)

    if columns == 'std':
        df = df.drop([5, 6, 7, 10, 11, 12, 13, 14, 15], 1)
        df.columns = ['Hm0', 'Tp', 'DirM']

    return df
