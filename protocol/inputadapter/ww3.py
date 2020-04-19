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
from input import ww3
from inputadapter.inputadapter import adapt_driver, adapt_all_drivers

driver_descriptors = {
    'wave': ['Hs', 'Tp', 'DirM'],
    'wind': ['VelV', 'DirV']
}


def read_data(file_name, path='.'):
    # Read the file
    data_path = os.path.join(path, file_name)
    data = ww3.read_time_series_file(data_path, columns='all')
    return data


def wave(file_name, metadata, path='.'):
    data = read_data(file_name, path)
    metadata['longitude'] = data.iloc[0, :]['lon']
    metadata['longitude'] = data.iloc[0, :]['lat']
    modf = adapt_driver(data, 'wave', driver_descriptors, metadata)

    return modf


def wind(file_name, metadata, path='.'):
    data = read_data(file_name, path)
    metadata['longitude'] = data.iloc[0, :]['lon']
    metadata['longitude'] = data.iloc[0, :]['lat']
    modf = adapt_driver(data, 'wind', driver_descriptors, metadata)

    return modf


def all_drivers(file_name, metadata, path='.'):
    data = read_data(file_name, path)
    modfs = adapt_all_drivers(data, driver_descriptors, metadata)

    return modfs
