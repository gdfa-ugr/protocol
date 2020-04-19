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
import fnmatch


def read_data(parent_folder, dams, name_folder, name_descriptor='Q', fill_value=0):
    """Read Rediam data

    Args:
        parent_folder (str): data folder root
        dams (list): dams
        name_folder (str): name of folder data
        name_descriptor (str, optional): name of driver descriptor
        fill_value (float, optional): fill NaN with this value

    Returns:
        Series: dams flow data
    """
    data = pd.Series()

    for dam in dams:
        data_dam = read_folder(os.path.join(parent_folder, dam), name_folder)
        data = data.add(data_dam, fill_value=fill_value)

    # Add name to series
    data.name = name_descriptor

    return data


def read_folder(parent_folder, name):
    """Read folder data

    Args:
        parent_folder (str): data folder root
        name (str): name of data variable

    Returns:
        Series: dam flow data
    """
    pattern = '*{}*'.format(name)

    matches = []
    for root, dirnames, file_names in os.walk(parent_folder):
        for file_name in fnmatch.filter(file_names, pattern):
            flow_data = read_file(os.path.join(root, file_name), name=name)
            matches.append(flow_data)

    data = pd.concat(matches)
    data = data.groupby(data.index).mean()  # average value with same index (assume NaN = 0)

    return data


def read_file(file_name, path='.', null_values=(-99.9, -99.99, -999, -9999, 990), name='values'):
    """Read file data

    Args:
        file_name (str): file name
        path (str, optional): path
        null_values (list, optional): considered null values
        name (str, optional): name of data variable

    Returns:
        Series: flow data
    """
    data = pd.read_csv(os.path.join(path, file_name), usecols=[1, 2], parse_dates={'datetime': [0]},
                       index_col=0, na_values=null_values, squeeze=True)
    data.rename(name, inplace=True)

    return data
