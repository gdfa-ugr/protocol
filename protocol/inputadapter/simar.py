#!/usr/bin/env python
# coding: utf-8
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

try:
    # noinspection PyUnresolvedReferences, PyCompatibility
    from builtins import *  # noqa
except ImportError:
    pass

from climate.read import simar
from inputadapter.inputadapter import adapt_driver, adapt_all_drivers

driver_descriptors = {
    'wave': ['Hm0', 'Tp', 'DirM'],
    'wind': ['VelV', 'DirV']
}


def read_data(file_name, path='.'):
    """ Read the data from SIMAR

    Args:
        file_name (str): file_name
        path (str): path

    Returns:
        tuple:

            * str: identifier of SIMAR file
            * DataFrame: wave and wind driver descriptors (Hm0, Tp, DirM, VelV, DirV)
    """
    # Read the file
    data = simar(file_name, path)

    return data


def wave(file_name, metadata, path='.'):
    """ Generate a MetOceanDF class with attributes for the driver WAVE from SIMAR

    Args:
        file_name (str): file_name
        metadata (dict): attributes of the MetOcean DataFrame
        path (str): path

    Returns:
        MetOceanDF: MetOceanDF class that includes the DataFrame with the drivers descriptors and the
        attributes
    """
    data = read_data(file_name, path)
    modf = adapt_driver(data[0], 'wave', driver_descriptors, metadata)

    return modf


def wind(file_name, metadata, path='.'):
    """ Generate a MetOceanDF class with attributes for the driver WIND from SIMAR

    Args:
        file_name (str): file_name
        metadata (dict): attributes of the MetOcean DataFrame
        path (str): path

    Returns:
        MetOceanDF: MetOceanDF class that includes the DataFrame with the drivers descriptors and the
        attributes
    """
    data = read_data(file_name, path)
    modf = adapt_driver(data[0], 'wind', driver_descriptors, metadata)

    return modf


def all_drivers(file_name, metadata, path='.'):
    """ Generate a MetOceanDF class with attributes for the drivers WAVE and WIND from SIMAR

    Args:
        file_name (str): file_name
        metadata (dict): attributes of the MetOcean DataFrame
        path (str): path

    Returns:
        list: List of MetOceanDF classes. One per driver
    """
    data = read_data(file_name, path)
    modfs = adapt_all_drivers(data[0], driver_descriptors, metadata)

    return modfs
