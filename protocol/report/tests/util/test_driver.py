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

import report.util
from metoceandataframe.metoceandataframe import MetOceanDF
from report import tests


def test_extract_main_descriptor_with_main_descriptor_name():
    # Input
    modf_file_name = 'guadalete_estuary_wave.modf'
    modf_path = os.path.join(tests.current_path, '..', '..', 'inputadapter',  'tests', 'output', 'modf')
    main_descriptor_name = 'DirM'
    # Read data
    path_name = os.path.join(modf_path, modf_file_name)
    modf = MetOceanDF.read_file(path_name)
    # Extract main descriptor
    main_descriptor = report.util.driver.extract_main_descriptor(modf, main_descriptor_name=main_descriptor_name)

    assert main_descriptor.name == main_descriptor_name


def test_extract_main_descriptor_without_main_descriptor_name():
    # Input
    modf_file_name = 'guadalete_estuary_wave.modf'
    modf_path = os.path.join(tests.current_path, '..', '..', 'inputadapter',  'tests', 'output', 'modf')
    # Read data
    path_name = os.path.join(modf_path, modf_file_name)
    modf = MetOceanDF.read_file(path_name)
    # Extract main descriptor
    main_descriptor = report.util.driver.extract_main_descriptor(modf)

    assert main_descriptor.name == 'Hm0'