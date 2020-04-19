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
import pandas as pd

# Build an absolute path from current script
current_path = os.path.abspath(os.path.dirname(__file__))
# Build relative paths from the former script path
sample_data_path = os.path.join(current_path, 'data')

sample_data_file = 'simar.msg'


def read_sample_simar(data_file=sample_data_file, data_path=sample_data_path, metadata=None):
    if metadata is None:
        metadata = {'source': 'SIMAR',
                    'id': 1052046,
                    'latitude': 36.5,
                    'longitude': -7.00,
                    'depth': 'deep_water'}

    simar = os.path.join(data_path, data_file)
    simar_data = pd.read_msgpack(simar)

    return metadata, simar_data
