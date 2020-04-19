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
from inputadapter import ww3, tests


def test_wave():
    file_name = 'Pto_5.txt'
    path = os.path.join(tests.full_data_path, 'ww3', 'Ptos')
    metadata = {'id': 'Pto_1'}

    modf_wave = ww3.wave(file_name, metadata, path)

    assert modf_wave.id == 'Pto_1'


def test_wind():
    file_name = 'Pto_5.txt'
    path = os.path.join(tests.full_data_path, 'ww3', 'Ptos')
    metadata = {'id': 'Pto_1'}

    modf_wind = ww3.wind(file_name, metadata, path)

    assert modf_wind.id == 'Pto_1'


def test_all_drivers():
    file_name = 'Pto_5.txt'
    path = os.path.join(tests.full_data_path, 'ww3', 'Ptos')
    metadata = {'id': 'Pto_1'}

    modfs = ww3.all_drivers(file_name, metadata, path)


def test_wave_cc_i():
    input_file_name = 'Pto_5.txt'
    input_path = os.path.join(tests.full_data_path, 'locations', 'cc', 'waves_wind', 'Ptos')
    output_csv_file_name = 'cancun_wave_I.csv'
    output_csv_path = os.path.join('output', 'csv')
    output_modf_file_name = 'cancun_wave_I.modf'
    output_modf_path = os.path.join('output', 'modf')
    metadata = {'source': 'wave_watch_III',
                'id': 'Pto_5',
                'depth': 'deep_water'}
    # Adapt driver
    modf_wave = ww3.wave(input_file_name, metadata, input_path)

    # Save results at output
    tests.save_to_csv(modf_wave, output_csv_file_name, output_csv_path)
    output_modf = os.path.join(output_modf_path, output_modf_file_name)
    modf_wave.to_file(output_modf)


def test_wave_cc_ii():
    input_file_name = 'Pto_7.txt'
    input_path = os.path.join(tests.full_data_path, 'locations', 'cc', 'waves_wind', 'Ptos')
    output_csv_file_name = 'cancun_wave_II.csv'
    output_csv_path = os.path.join('output', 'csv')
    output_modf_file_name = 'cancun_wave_II.modf'
    output_modf_path = os.path.join('output', 'modf')
    metadata = {'source': 'wave_watch_III',
                'id': 'Pto_7',
                'depth': 'deep_water'}
    # Adapt driver
    modf_wave = ww3.wave(input_file_name, metadata, input_path)

    # Save results at output
    tests.save_to_csv(modf_wave, output_csv_file_name, output_csv_path)
    output_modf = os.path.join(output_modf_path, output_modf_file_name)
    modf_wave.to_file(output_modf)


def test_wind_cc_i():
    input_file_name = 'Pto_5.txt'
    input_path = os.path.join(tests.full_data_path, 'locations', 'cc', 'waves_wind', 'Ptos')
    output_csv_file_name = 'cancun_wind_I.csv'
    output_csv_path = os.path.join('output', 'csv')
    output_modf_file_name = 'cancun_wind_I.modf'
    output_modf_path = os.path.join('output', 'modf')
    metadata = {'source': 'wave_watch_III',
                'id': 'Pto_5',
                'depth': 'deep_water'}
    # Adapt driver
    modf_wind = ww3.wind(input_file_name, metadata, input_path)

    # Save results at output
    tests.save_to_csv(modf_wind, output_csv_file_name, output_csv_path)
    output_modf = os.path.join(output_modf_path, output_modf_file_name)
    modf_wind.to_file(output_modf)


def test_wind_cc_ii():
    input_file_name = 'Pto_7.txt'
    input_path = os.path.join(tests.full_data_path, 'locations', 'cc', 'waves_wind', 'Ptos')
    output_csv_file_name = 'cancun_wind_II.csv'
    output_csv_path = os.path.join('output', 'csv')
    output_modf_file_name = 'cancun_wind_II.modf'
    output_modf_path = os.path.join('output', 'modf')
    metadata = {'source': 'wave_watch_III',
                'id': 'Pto_7',
                'depth': 'deep_water'}
    # Adapt driver
    modf_wind = ww3.wind(input_file_name, metadata, input_path)

    # Save results at output
    tests.save_to_csv(modf_wind, output_csv_file_name, output_csv_path)
    output_modf = os.path.join(output_modf_path, output_modf_file_name)
    modf_wind.to_file(output_modf)
