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


def read_time_series_file(data_path, name_descriptor='Eta', path='.'):
    # Read the file
    raw_data = pd.read_table(os.path.join(path, data_path), header=None, skiprows=6, delim_whitespace=True,
                             parse_dates=[[0, 1]], index_col=0)

    # Round the dates to hours
    t = raw_data.index.round(freq='H')
    raw_data.index = t

    # Add header to dataframe
    raw_data.columns = [name_descriptor]

    return raw_data
