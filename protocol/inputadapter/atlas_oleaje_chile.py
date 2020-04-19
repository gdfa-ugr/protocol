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
from input import atlas_oleaje_chile
from inputadapter.inputadapter import adapt_driver, adapt_all_drivers

driver_descriptors = {
    'wave': ['Hm0', 'Tp', 'DirM'],
}


def read_data(file_name, path='.'):
    # Read the file
    data_path = os.path.join(path, file_name)
    data = atlas_oleaje_chile.read_time_series_file(data_path, columns='std')
    return data


def wave(file_name, metadata, path='.'):
    data = read_data(file_name, path)
    modf = adapt_driver(data, 'wave', driver_descriptors, metadata)

    return modf


def all_drivers(file_name, metadata, path='.'):
    data = read_data(file_name, path)
    modfs = adapt_all_drivers(data, driver_descriptors, metadata)

    return modfs
