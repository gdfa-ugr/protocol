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

from climate import read
from climate.util.os import make_folder

# Build an absolute path from current script
current_path = os.path.abspath(os.path.dirname(__file__))
# Build relative paths from the former script path
sample_data_path = os.path.join(current_path, 'data')
full_data_path = os.path.join(current_path, '..', '..', '..', '..', 'data')
output_img_path = os.path.join(current_path, 'output', 'img')

sample_data_file = 'simar.txt'
full_data_file = os.path.join('simar', 'SIMAR_1052046')
sample_preprocessed_file = 'full_simar_preprocessed.msg'


def read_sample_simar(data_file=sample_data_file, data_path=sample_data_path):
    # noinspection PyTypeChecker
    data_simar, code = read.simar(data_file, data_path)

    return data_simar


def read_full_simar(data_file=full_data_file, data_path=full_data_path):
    # noinspection PyTypeChecker
    data_simar, code = read.simar(data_file, data_path)

    return data_simar


def read_sample_preprocessed_simar(data_file=sample_preprocessed_file, data_path=sample_data_path):
    # noinspection PyTypeChecker
    data_simar = pd.read_msgpack(os.path.join(data_path, data_file))

    return data_simar


def get_img_path(file_name):
    make_folder(output_img_path)
    img_path = os.path.join(output_img_path, file_name)

    return img_path
