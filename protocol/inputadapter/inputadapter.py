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
from metoceandataframe import metoceandataframe


def adapt_driver(data, driver, driver_descriptors, metadata):
    """ Read the DataFrame from file and generate a MetOceanDF class with attributes

    Args:
        data (pd.DataFrame): data
        driver (str): driver
        driver_descriptors (dict): dictionary with the descriptors associated to each driver
        metadata (dict): Attributes associated to each driver

    Returns:
        MetOceanDF: MetOceanDF class that includes the DataFrame with the drivers descriptors and the
        attributes
    """
    if isinstance(data, pd.Series):
        data = data.to_frame()

    metadata['driver'] = driver

    # Extract columns
    columns = driver_descriptors[metadata['driver']]

    data_driver = data.loc[:, columns]

    # Create class MetOcean DataFrame
    modf = metoceandataframe.MetOceanDF(data_driver)

    # Assign atributes
    for attribute in modf.get_properties().keys():
        setattr(modf, attribute, metadata.get(attribute))

    return modf


def adapt_all_drivers(data, driver_descriptors, metadata):
    """ Generate a list MetOceanDF class, one per driver, with their associated attributes
    Args:
        data (pd.DataFrame): data
        driver_descriptors (dict): dictionary with the descriptors associated to each driver
        metadata (dict): Attributes associated to each driver

    Returns:
        list: list of MetOceanDF class, one per driver
    """
    data_drivers = []
    for descriptor in driver_descriptors.keys():
        data_driver = adapt_driver(data, descriptor, driver_descriptors, metadata)
        data_drivers.append(data_driver)

    return data_drivers
