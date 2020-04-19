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

# Build an absolute path from current script
current_path = os.path.abspath(os.path.dirname(__file__))
# Build relative paths from the former script path
sample_data_path = os.path.join(current_path, 'data')
full_data_path = os.path.join(current_path, '..', '..', '..', '..', 'data')