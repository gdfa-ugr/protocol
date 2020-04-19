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

from input import rediam, tests


def test_read_data():
    name = 'caudal_desembalsado'
    dams = ['270_BORNOS', '273_GUADALCACIN']

    data_path = os.path.join(tests.full_data_path, 'rediam')

    # noinspection PyTypeChecker
    data = rediam.read_data(data_path, dams, name)

    assert data.shape[0] == 157368


def test_read_folder():
    name = 'caudal_desembalsado'
    parent_folder = os.path.join(tests.sample_data_path, 'rediam', 'dam')

    # noinspection PyTypeChecker
    data_flow_dam = rediam.read_folder(parent_folder, name)

    assert data_flow_dam.name == name
    assert data_flow_dam.shape[0] == 17496
    assert data_flow_dam.loc['2001-09-29 07:00:00'] == 0.64


def test_read_file():
    data_file = 'flow.csv'
    data_path = os.path.join(tests.sample_data_path, 'rediam')

    # noinspection PyTypeChecker
    data_flow = rediam.read_file(data_file, path=data_path)

    assert data_flow.shape[0] == 8737
    assert data_flow.ix[153, 'valor'] == 0.001
