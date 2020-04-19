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


def read_file(data_path, name_descriptor='Q', path='.'):
    df = pd.read_table(os.path.join(path, data_path), header=None, delim_whitespace=True)
    date_col = dates.extract_date(df.iloc[:, 0:4])
    df.index = date_col
    df.drop(df.columns[0:4], axis=1, inplace=True)

    # Add header to dataframe
    df.columns = [name_descriptor]

    return df
