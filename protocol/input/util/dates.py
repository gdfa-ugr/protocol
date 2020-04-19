#!/usr/bin/env python
# coding: utf-8
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import numpy as np

try:
    # noinspection PyUnresolvedReferences, PyCompatibility
    from builtins import *  # noqa
except ImportError:
    pass

import pandas as pd


def extract_date(df):
    """Extract the date from the columns of ``df``.

    Args:
        df: DataFrame with the date-time columns related

    Returns:
        DateTime object
    """

    time = ''

    if df.shape[1] > 2:
        for col in df.iloc[:, 0:2]:
            time += pd.to_numeric(df.loc[:, col], downcast='integer').astype('str') + '-'
        time += pd.to_numeric(df.iloc[:, 2], downcast='integer').astype('str')

        if df.shape[1] > 3:
            time += ' '
            for col in df.iloc[:, 3:-1]:
                time += pd.to_numeric(df.loc[:, col], downcast='integer').astype('str') + ':'
            time += pd.to_numeric(df.iloc[:, -1], downcast='integer').astype('str')
            if df.shape[1] == 4:
                time += ':0'

    return pd.to_datetime(time)


def convert_datenum_datetime(dates):
    seconds = dates * 24 * 60 * 60
    origin = np.datetime64('1970-01-01T00:00:00') - np.datetime64('0000-01-01T00:00:00') + np.timedelta64(1, 'D')

    datetime = seconds.astype('datetime64[s]') - origin

    return datetime
