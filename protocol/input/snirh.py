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

from input.util.dates import extract_date


def read_file(file_name, path='.', name_descriptor='Q'):
    data_path = os.path.join(path, file_name)
    data = pd.read_csv(data_path, header=None, delimiter='/| |:|;')
    date_col = extract_date(data.iloc[:, [2, 1, 0]])
    serie = data.iloc[:, 7]
    serie.index = date_col
    serie.name = name_descriptor

    return serie
