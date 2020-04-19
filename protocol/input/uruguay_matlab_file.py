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

from input.util import matlab


def read_file(data_path, variables, time_column, path='.'):
    data = matlab.read_time_series_matlab(data_path=os.path.join(path, data_path), variables=variables,
                                          time_column_pos=time_column)

    return data
