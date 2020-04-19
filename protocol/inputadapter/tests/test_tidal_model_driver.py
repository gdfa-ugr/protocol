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
from inputadapter import tidal_model_driver, tests


def test_tidal_model_driver():
    file_name = 'time_series.out'
    metadata = {'id': 'Guadalete',
                'latitude': 36.45,
                'longitude': 353.45}
    path = os.path.join(tests.sample_data_path, 'tidal_model_driver')

    modf_astron_tide = tidal_model_driver.astronomical_tide(file_name, metadata, path)

    assert modf_astron_tide.id == 'Guadalete'


def test_all_drivers():
    file_name = 'time_series.out'
    metadata = {'id': 'Guadalete',
                'latitude': 36.45,
                'longitude': 353.45}
    path = os.path.join(tests.sample_data_path, 'tidal_model_driver')

    modfs = tidal_model_driver.astronomical_tide(file_name, metadata, path)


def test_tidal_model_driver_eg():
    input_file_name = 'time_series.out'
    input_path = os.path.join(tests.full_data_path, 'locations', 'eg', 'astronomical_tide')
    output_csv_file_name = 'guadalete_estuary_astronomical_tide.csv'
    output_csv_path = os.path.join('output', 'csv')
    output_modf_file_name = 'guadalete_estuary_astronomical_tide.modf'
    output_modf_path = os.path.join('output', 'modf')

    metadata = {'source': 'tidal_model_driver',
                'latitude': 36.45,
                'longitude': -6.55,
                'depth': 'deep_water'}
    # Adapt driver
    modf_astron_tide = tidal_model_driver.astronomical_tide(input_file_name, metadata, input_path)

    # Save results at output
    tests.save_to_csv(modf_astron_tide, output_csv_file_name, output_csv_path)
    output_modf = os.path.join(output_modf_path, output_modf_file_name)
    modf_astron_tide.to_file(output_modf)


def test_tidal_model_driver_pg():
    input_file_name = 'time_series.out'
    input_path = os.path.join(tests.full_data_path, 'locations', 'pg', 'astronomical_tide')
    output_csv_file_name = 'granada_beach_astronomical_tide.csv'
    output_csv_path = os.path.join('output', 'csv')
    output_modf_file_name = 'granada_beach_astronomical_tide.modf'
    output_modf_path = os.path.join('output', 'modf')
    metadata = {'source': 'tidal_model_driver',
                'latitude': 36.65,
                'longitude': -3.55,
                'depth': 'deep_water'}
    # Adapt driver
    modf_astron_tide = tidal_model_driver.astronomical_tide(input_file_name, metadata, input_path)

    # Save results at output
    tests.save_to_csv(modf_astron_tide, output_csv_file_name, output_csv_path)
    output_modf = os.path.join(output_modf_path, output_modf_file_name)
    modf_astron_tide.to_file(output_modf)


def test_tidal_model_driver_em():
    input_file_name = 'time_series.out'
    input_path = os.path.join(tests.full_data_path, 'locations', 'em', 'astronomical_tide')
    output_csv_file_name = 'mondego_estuary_astronomical_tide.csv'
    output_csv_path = os.path.join('output', 'csv')
    output_modf_file_name = 'mondego_estuary_astronomical_tide.modf'
    output_modf_path = os.path.join('output', 'modf')
    metadata = {'source': 'tidal_model_driver',
                'latitude': 40.14,
                'longitude': -9,
                'depth': 'deep_water'}
    # Adapt driver
    modf_astron_tide = tidal_model_driver.astronomical_tide(input_file_name, metadata, input_path)

    # Save results at output
    tests.save_to_csv(modf_astron_tide, output_csv_file_name, output_csv_path)
    output_modf = os.path.join(output_modf_path, output_modf_file_name)
    modf_astron_tide.to_file(output_modf)


def test_tidal_model_driver_jl():
    input_file_name = 'time_series.out'
    input_path = os.path.join(tests.full_data_path, 'locations', 'jl', 'astronomical_tide')
    output_csv_file_name = 'juan_lacaze_astronomical_tide.csv'
    output_csv_path = os.path.join('output', 'csv')
    output_modf_file_name = 'juan_lacaze_astronomical_tide.modf'
    output_modf_path = os.path.join('output', 'modf')
    metadata = {'source': 'tidal_model_driver',
                'latitude': -34.49,
                'longitude': -57.38,
                'depth': 'deep_water'}
    # Adapt driver
    modf_astron_tide = tidal_model_driver.astronomical_tide(input_file_name, metadata, input_path)

    # Save results at output
    tests.save_to_csv(modf_astron_tide, output_csv_file_name, output_csv_path)
    output_modf = os.path.join(output_modf_path, output_modf_file_name)
    modf_astron_tide.to_file(output_modf)


def test_tidal_model_driver_cc():
    input_file_name = 'time_series.out'
    input_path = os.path.join(tests.full_data_path, 'locations', 'cc', 'astronomical_tide')
    output_csv_file_name = 'cancun_astronomical_tide.csv'
    output_csv_path = os.path.join('output', 'csv')
    output_modf_file_name = 'cancun_astronomical_tide.modf'
    output_modf_path = os.path.join('output', 'modf')
    metadata = {'source': 'tidal_model_driver',
                'latitude': 21.13,
                'longitude': -86.61,
                'depth': 'deep_water'}
    # Adapt driver
    modf_astron_tide = tidal_model_driver.astronomical_tide(input_file_name, metadata, input_path)

    # Save results at output
    tests.save_to_csv(modf_astron_tide, output_csv_file_name, output_csv_path)
    output_modf = os.path.join(output_modf_path, output_modf_file_name)
    modf_astron_tide.to_file(output_modf)


def test_tidal_model_driver_vp():
    input_file_name = 'time_series.out'
    input_path = os.path.join(tests.full_data_path, 'locations', 'vp', 'astronomical_tide')
    output_csv_file_name = 'gran_valparaiso_astronomical_tide.csv'
    output_csv_path = os.path.join('output', 'csv')
    output_modf_file_name = 'gran_valparaiso_astronomical_tide.modf'
    output_modf_path = os.path.join('output', 'modf')
    metadata = {'source': 'tidal_model_driver',
                'latitude': -32.94,
                'longitude': -71.65,
                'depth': 'deep_water'}
    # Adapt driver
    modf_astron_tide = tidal_model_driver.astronomical_tide(input_file_name, metadata, input_path)

    # Save results at output
    tests.save_to_csv(modf_astron_tide, output_csv_file_name, output_csv_path)
    output_modf = os.path.join(output_modf_path, output_modf_file_name)
    modf_astron_tide.to_file(output_modf)
