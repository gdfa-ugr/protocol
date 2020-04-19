#!/usr/bin/env python
# coding: utf-8
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

try:
    # noinspection PyUnresolvedReferences, PyCompatibility
    from builtins import *  # noqa
except ImportError:
    pass


from input import rediam
from inputadapter.inputadapter import adapt_driver, adapt_all_drivers

driver_descriptors = {
    'river_discharge': ['Q']}


def read_data(file_name, dams, path='.'):
    # Read the file
    data = rediam.read_data(path, dams, file_name)

    return data


def river_discharge(file_name, metadata, dams, path):
    data = read_data(file_name, dams, path)
    modf = adapt_driver(data, 'river_discharge', driver_descriptors, metadata)

    return modf


def all_drivers(file_name, metadata, dams, path='.'):
    data = read_data(file_name, dams, path)
    modfs = adapt_all_drivers(data, driver_descriptors, metadata)

    return modfs
