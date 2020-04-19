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
from inputadapter import uruguay_matlab_file, tests


def test_wave():
    file_name = 'Oleaje.mat'
    metadata = {'id': 'Uruguay',
                'latitude': None,
                'longitude': None}
    path = os.path.join(tests.full_data_path, 'matlab')

    modf_wave = uruguay_matlab_file.wave(file_name, 'JL1cor', 0, metadata, path)

    assert modf_wave.id == 'Uruguay'


def test_all_drivers():
    file_name = 'Oleaje.mat'
    metadata = {'id': 'Uruguay',
                'latitude': None,
                'longitude': None}
    path = os.path.join(tests.full_data_path, 'matlab')

    modfs = uruguay_matlab_file.all_drivers(file_name, 'JL1cor', 0, metadata, path)


def test_wave_jl():
    input_file_name = 'Oleaje.mat'
    input_path = os.path.join(tests.full_data_path, 'locations', 'jl', 'waves_wind')
    output_csv_file_name = 'juan_lacaze_wave.csv'
    output_csv_path = os.path.join('output', 'csv')
    output_modf_file_name = 'juan_lacaze_wave.modf'
    output_modf_path = os.path.join('output', 'modf')
    metadata = {'source': 'Unknown',
                'latitude': -34.47,
                'longitude': -57.45,
                'depth': 5.5}
    # Adapt driver
    modf_wave = uruguay_matlab_file.wave(input_file_name, 'JL1cor', 0, metadata, input_path)

    # Save results at output
    tests.save_to_csv(modf_wave, output_csv_file_name, output_csv_path)
    output_modf = os.path.join(output_modf_path, output_modf_file_name)
    modf_wave.to_file(output_modf)
