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
from inputadapter import atlas_oleaje_chile, tests


def test_wave():
    file_name = 'wave.txt'
    path = os.path.join(tests.sample_data_path, 'atlas_oleaje_chile')
    metadata = {'id': 'Nodo_8'}

    modf_wave = atlas_oleaje_chile.wave(file_name, metadata, path)

    assert modf_wave.id == 'Nodo_8'


def test_all_drivers():
    file_name = 'wave.txt'
    path = os.path.join(tests.sample_data_path, 'atlas_oleaje_chile')
    metadata = {'id': 'Nodo_8'}

    modfs = atlas_oleaje_chile.all_drivers(file_name, metadata, path)


def test_wave_vp():
    input_file_name = 'Nodo 8 (-33,-73) - Valparaiso.txt'
    input_path = os.path.join(tests.full_data_path, 'locations', 'vp', 'waves_wind')
    output_csv_file_name = 'gran_valparaiso_wave.csv'
    output_csv_path = os.path.join('output', 'csv')
    output_modf_file_name = 'gran_valparaiso_wave.modf'
    output_modf_path = os.path.join('output', 'modf')
    metadata = {'source': 'Atlas_Chile',
                'id': 'Nodo_8',
                'latitude': -33.00,
                'longitude': -73.00,
                'depth': 'Deep water'}
    # Adapt driver
    modf_wave = atlas_oleaje_chile.wave(input_file_name, metadata, input_path)

    # Save results at output
    tests.save_to_csv(modf_wave, output_csv_file_name, output_csv_path)
    output_modf = os.path.join(output_modf_path, output_modf_file_name)
    modf_wave.to_file(output_modf)
