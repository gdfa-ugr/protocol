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

from metoceandataframe import tests
from metoceandataframe.metoceandataframe import MetOceanDF


def test_constructor():
    metadata, simar_data = tests.read_sample_simar()
    modf = MetOceanDF(simar_data, metadata=metadata)

    assert modf.latitude == 36.5


def test_get_properties():
    metadata, simar_data = tests.read_sample_simar()
    modf = MetOceanDF(simar_data, metadata=metadata)

    properties = modf.get_properties()

    assert properties['latitude'] == 36.5


def test_get_dataframe():
    metadata, simar_data = tests.read_sample_simar()
    modf = MetOceanDF(simar_data, metadata=metadata)

    df = modf.get_dataframe()

    assert df.iloc[4]['Hm0'] == 2.1


def test_to_file():
    metadata, simar_data = tests.read_sample_simar()
    modf = MetOceanDF(simar_data, metadata=metadata)

    output_name = 'simar.modf'
    output_path = os.path.join(tests.current_path, 'output', 'modf')

    modf.to_file(output_name, output_path)


def test_read_file():
    output_name = 'simar.modf'
    output_path = os.path.join(tests.current_path, 'output', 'modf')

    modf = MetOceanDF.read_file(output_path, output_name)

    assert modf.latitude == 36.5
