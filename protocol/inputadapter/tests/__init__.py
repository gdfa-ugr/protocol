#!/usr/bin/env python
# coding: utf-8
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

try:
    # noinspection PyUnresolvedReferences, PyCompatibility
    from builtins import *  # noqa
except ImportError:
    pass

from future.utils import bytes_to_native_str as n  # to specify str type on both Python 2/3

import os

# Build an absolute path from current script
current_path = os.path.abspath(os.path.dirname(__file__))
# Build relative paths from the former script path
sample_data_path = os.path.join(current_path, 'data')
full_data_path = os.path.join(current_path, '..', '..', '..', '..', 'data')


def save_to_csv(data, file_name, path='.'):
    path_name = os.path.join(path, file_name)
    data.to_csv(path_name, sep=n(b'\t'))


