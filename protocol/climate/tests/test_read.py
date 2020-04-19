#!/usr/bin/env python
# coding: utf-8
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

try:
    # noinspection PyUnresolvedReferences, PyCompatibility
    from builtins import *  # noqa
except ImportError:
    pass

from climate import read, tests


def test_simar_code():
    _, code = read.simar(tests.sample_data_file, tests.sample_data_path)

    assert code == '1052046'


def test_simar_data():
    data_simar, _ = read.simar(tests.sample_data_file, tests.sample_data_path)

    assert data_simar.shape[0] == 216
    assert data_simar.ix[153, 'Tp'] == 11.2
