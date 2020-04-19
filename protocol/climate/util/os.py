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
import errno


def make_folder(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def save_to_msgpack(data, file_name, path='.'):
    folder = os.path.join(path, file_name)
    data.to_msgpack(folder, compress=n(b'zlib'))

    return

