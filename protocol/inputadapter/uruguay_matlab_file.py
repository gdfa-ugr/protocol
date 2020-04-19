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
from input import uruguay_matlab_file
from inputadapter.inputadapter import adapt_driver, adapt_all_drivers

driver_descriptors = {
    'wave': ['Hs', 'Tp', 'DirM']}


def read_data(file_name, variable_name, time_column_pos, path='.'):
    # Read the file
    data_path = os.path.join(path, file_name)
    data = uruguay_matlab_file.read_file(data_path, variables=variable_name, time_column=time_column_pos)
    data.columns = ['t', 'Hs', 'Tp', 'DirM']

    return data


def wave(file_name, variable_name, time_column_pos, metadata, path='.'):
    data = read_data(file_name, variable_name, time_column_pos, path)
    modf = adapt_driver(data, 'wave', driver_descriptors, metadata)

    return modf


def all_drivers(file_name, variable_name, time_column_pos, metadata, path='.'):
    data = read_data(file_name, variable_name, time_column_pos, path)
    modfs = adapt_all_drivers(data, driver_descriptors, metadata)

    return modfs
