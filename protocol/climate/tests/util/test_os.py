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
from climate import read, tests
from preprocessing import missing_values
from climate.util.os import save_to_msgpack


def test_save_to_msgpack():

    data_file = os.path.join('simar', 'SIMAR_1052046')
    file_name = 'full_simar_preprocessed.msg'
    path = 'D:\\REPOSITORIO GIT\\protocol_project\\data\\intermediate_files'

    # Read simar
    data_simar, _ = read.simar(data_file, tests.full_data_path)

    # Preproccesing simar
    time_step = missing_values.find_timestep(data_simar, n_random_values=10)
    data_clean = missing_values.erase_null_values(data_simar, method='all')
    data_simar_interp = missing_values.fill_missing_values(data_clean, time_step, technique='interpolation',
                                                           method='nearest', limit=720, limit_direction='both')
    # Check missing values
    miss_values = missing_values.find_missing_values(data_simar_interp, time_step)

    # Save simar
    if miss_values.empty:
        save_to_msgpack(data_simar_interp, file_name, path)

    return
