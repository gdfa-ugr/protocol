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

from input import era
from inputadapter.inputadapter import adapt_driver, adapt_all_drivers


driver_descriptors = {
    'sea_level_pressure': ['Slp']}


def read_data(file_name, metadata, path='.'):
    # Read the file
    data_path = os.path.join(path, file_name)
    (data, units, coordinates) = era.read_netcdf_file(data_path)
    metadata['latitude'] = coordinates['lat']
    metadata['longitude'] = coordinates['lon']
    return data, units, metadata


def sea_level_pressure(file_name, metadata, path='.'):
    (data, units, metadata) = read_data(file_name, metadata, path)
    modf = adapt_driver(data, 'sea_level_pressure', driver_descriptors, metadata)

    return modf


def all_drivers(file_name, metadata, path='.'):
    (data, units, metadata) = read_data(file_name, metadata, path)
    modfs = adapt_all_drivers(data, driver_descriptors, metadata)

    return modfs
