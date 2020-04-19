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
from input import saih
from inputadapter.inputadapter import adapt_driver, adapt_all_drivers

driver_descriptors = {
    'river_discharge': ['Q']}


def read_data(file_name, path='.'):
    # Read the file
    data_path = os.path.join(path, file_name)
    data = saih.read_file(data_path)

    return data


def river_discharge(file_name, metadata, path='.'):
    data = read_data(file_name, path)
    modf = adapt_driver(data, 'river_discharge', driver_descriptors, metadata)

    return modf


def all_drivers(file_name, metadata, path='.'):
    data = read_data(file_name, path)
    modfs = adapt_all_drivers(data, driver_descriptors, metadata)

    return modfs
