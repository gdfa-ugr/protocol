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
from inputadapter import simar, tests


def test_wave():
    file_name = 'simar.txt'
    path = os.path.join(tests.sample_data_path, 'simar')
    metadata = {'id': 1052046,
                'latitude': 36.5,
                'longitude': -7.00}
    modf_wave = simar.wave(file_name, metadata, path)

    assert modf_wave.id == 1052046
    assert modf_wave.latitude == 36.5


def test_wind():
    file_name = 'simar.txt'
    path = os.path.join(tests.sample_data_path, 'simar')
    metadata = {'id': 1052046,
                'latitude': 36.5,
                'longitude': -7.00}
    modf_wind = simar.wind(file_name, metadata, path)

    assert modf_wind.id == 1052046
    assert modf_wind.latitude == 36.5

    return True


def test_all_drivers():
    file_name = 'simar.txt'
    path = os.path.join(tests.sample_data_path, 'simar')
    metadata = {'id': 1052046,
                'latitude': 36.5,
                'longitude': -7.00}
    modfs = simar.all_drivers(file_name, metadata, path)


def test_wave_eg():
    input_file_name = 'SIMAR_1052046'
    input_path = os.path.join(tests.full_data_path, 'locations', 'eg', 'waves_wind')
    output_csv_file_name = 'guadalete_estuary_wave.csv'
    output_csv_path = os.path.join('output', 'csv')
    output_modf_file_name = 'guadalete_estuary_wave.modf'
    output_modf_path = os.path.join('output', 'modf')
    metadata = {'source': 'SIMAR',
                'id': 1052046,
                'latitude': 36.5,
                'longitude': -7.00,
                'depth': 'deep_water'}
    # Adapt driver
    modf_wave = simar.wave(input_file_name, metadata, input_path)

    # Save results at output
    tests.save_to_csv(modf_wave, output_csv_file_name, output_csv_path)
    output_modf = os.path.join(output_modf_path, output_modf_file_name)
    modf_wave.to_file(output_modf)


def test_wind_eg():
    input_file_name = 'SIMAR_1052046'
    input_path = os.path.join(tests.full_data_path, 'locations', 'eg', 'waves_wind')
    output_csv_file_name = 'guadalete_estuary_wind.csv'
    output_csv_path = os.path.join('output', 'csv')
    output_modf_file_name = 'guadalete_estuary_wind.modf'
    output_modf_path = os.path.join('output', 'modf')
    metadata = {'source': 'SIMAR',
                'id': 1052046,
                'latitude': 36.5,
                'longitude': -7.00,
                'depth': 'deep_water'}
    # Adapt driver
    modf_wind = simar.wind(input_file_name, metadata, input_path)

    # Save results at output
    tests.save_to_csv(modf_wind, output_csv_file_name, output_csv_path)
    output_modf = os.path.join(output_modf_path, output_modf_file_name)
    modf_wind.to_file(output_modf)


def test_wave_pg():
    input_file_name = 'SIMAR_2041080'
    input_path = os.path.join(tests.full_data_path, 'locations', 'pg', 'waves_wind')
    output_csv_file_name = 'granada_beach_wave.csv'
    output_csv_path = os.path.join('output', 'csv')
    output_modf_file_name = 'granada_beach_wave.modf'
    output_modf_path = os.path.join('output', 'modf')
    metadata = {'source': 'SIMAR',
                'id': 2041080,
                'latitude': 36.667,
                'longitude': -3.583,
                'depth': 'deep_water'}
    # Adapt driver
    modf_wave = simar.wave(input_file_name, metadata, input_path)

    # Save results at output
    tests.save_to_csv(modf_wave, output_csv_file_name, output_csv_path)
    output_modf = os.path.join(output_modf_path, output_modf_file_name)
    modf_wave.to_file(output_modf)


def test_wind_pg():
    input_file_name = 'SIMAR_2041080'
    input_path = os.path.join(tests.full_data_path, 'locations', 'pg', 'waves_wind')
    output_csv_file_name = 'granada_beach_wind.csv'
    output_csv_path = os.path.join('output', 'csv')
    output_modf_file_name = 'granada_beach_wind.modf'
    output_modf_path = os.path.join('output', 'modf')
    metadata = {'source': 'SIMAR',
                'id': 2041080,
                'latitude': 36.667,
                'longitude': -3.583,
                'depth': 'deep_water'}
    # Adapt driver
    modf_wind = simar.wind(input_file_name, metadata, input_path)

    # Save results at output
    tests.save_to_csv(modf_wind, output_csv_file_name, output_csv_path)
    output_modf = os.path.join(output_modf_path, output_modf_file_name)
    modf_wind.to_file(output_modf)


def test_wave_em_i():
    input_file_name = 'SIMAR_1042060'
    input_path = os.path.join(tests.full_data_path, 'locations', 'em', 'waves_wind')
    output_csv_file_name = 'mondego_estuary_wave_I.csv'
    output_csv_path = os.path.join('output', 'csv')
    output_modf_file_name = 'mondego_estuary_wave_I.modf'
    output_modf_path = os.path.join('output', 'modf')
    metadata = {'source': 'SIMAR',
                'id': 1042060,
                'latitude': 40.000,
                'longitude': -9.500,
                'depth': 'deep_water'}
    # Adapt driver
    modf_wave = simar.wave(input_file_name, metadata, input_path)

    # Save results at output
    tests.save_to_csv(modf_wave, output_csv_file_name, output_csv_path)
    output_modf = os.path.join(output_modf_path, output_modf_file_name)
    modf_wave.to_file(output_modf)


def test_wind_em_i():
    input_file_name = 'SIMAR_1042060'
    input_path = os.path.join(tests.full_data_path, 'locations', 'em', 'waves_wind')
    output_csv_file_name = 'mondego_estuary_wind_I.csv'
    output_csv_path = os.path.join('output', 'csv')
    output_modf_file_name = 'mondego_estuary_wind_I.modf'
    output_modf_path = os.path.join('output', 'modf')
    metadata = {'source': 'SIMAR',
                'id': 1042060,
                'latitude': 40.000,
                'longitude': -9.500,
                'depth': 'deep_water'}
    # Adapt driver
    modf_wind = simar.wind(input_file_name, metadata, input_path)

    # Save results at output
    tests.save_to_csv(modf_wind, output_csv_file_name, output_csv_path)
    output_modf = os.path.join(output_modf_path, output_modf_file_name)
    modf_wind.to_file(output_modf)


def test_wave_em_ii():
    input_file_name = 'SIMAR_1042062'
    input_path = os.path.join(tests.full_data_path, 'locations', 'em', 'waves_wind')
    output_csv_file_name = 'mondego_estuary_wave_II.csv'
    output_csv_path = os.path.join('output', 'csv')
    output_modf_file_name = 'mondego_estuary_wave_II.modf'
    output_modf_path = os.path.join('output', 'modf')
    metadata = {'source': 'SIMAR',
                'id': 1042062,
                'latitude': 40.000,
                'longitude': -9.500,
                'depth': 'deep_water'}
    # Adapt driver
    modf_wave = simar.wave(input_file_name, metadata, input_path)

    # Save results at output
    tests.save_to_csv(modf_wave, output_csv_file_name, output_csv_path)
    output_modf = os.path.join(output_modf_path, output_modf_file_name)
    modf_wave.to_file(output_modf)


def test_wind_em_ii():
    input_file_name = 'SIMAR_1042062'
    input_path = os.path.join(tests.full_data_path, 'locations', 'em', 'waves_wind')
    output_csv_file_name = 'mondego_estuary_wind_II.csv'
    output_csv_path = os.path.join('output', 'csv')
    output_modf_file_name = 'mondego_estuary_wind_II.modf'
    output_modf_path = os.path.join('output', 'modf')
    metadata = {'source': 'SIMAR',
                'id': 1042062,
                'latitude': 40.000,
                'longitude': -9.500,
                'depth': 'deep_water'}
    # Adapt driver
    modf_wind = simar.wind(input_file_name, metadata, input_path)

    # Save results at output
    tests.save_to_csv(modf_wind, output_csv_file_name, output_csv_path)
    output_modf = os.path.join(output_modf_path, output_modf_file_name)
    modf_wind.to_file(output_modf)
