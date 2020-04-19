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
import matplotlib.pyplot as plt
from inputadapter import era, tests


def test_all_drivers():
    # Read the first file (era interim until 1979)
    file_name = 'data.nc'
    folder = 'era_interim'
    path = os.path.join(tests.sample_data_path, 'era', folder)
    metadata = {'id': 'Guadalete'}
    modf_slp_era_interim = era.all_drivers(file_name, metadata, path)

    # Read the second file (era 40 from 1979)
    file_name = 'data.nc'
    folder = 'era_40'
    path = os.path.join(tests.sample_data_path, 'era', folder)
    metadata = {'id': 'Guadalete'}
    modf_slp_era_40 = era.all_drivers(file_name, metadata, path)

    # Combination of both files
    modf_slp = modf_slp_era_interim[0].combine_first(modf_slp_era_40[0])

    assert modf_slp.latitude[0] == 36.625
    assert modf_slp.longitude[0] == 353.625


def test_sea_level_pressure_eg():
    # Read the first file (era interim until 1979)
    input_file_name = 'data.nc'
    input_path = os.path.join(tests.full_data_path, 'locations', 'eg', 'sea_level_pressure', 'ERA_cadiz')
    metadata = {'source': 'ERA(ECMWF)'}
    modf_slp_era_interim = era.sea_level_pressure(input_file_name, metadata, input_path)

    # Read the second file (era 40 from 1979)
    input_path = os.path.join(tests.full_data_path, 'locations', 'eg', 'sea_level_pressure', 'ERA40_cadiz')
    metadata = {'source': 'ERA(ECMWF)'}
    modf_slp_era_40 = era.sea_level_pressure(input_file_name, metadata, input_path)

    # Combination of both files
    modf_slp = modf_slp_era_interim.combine_first(modf_slp_era_40)

    # Plot results
    plt.figure()
    plt.plot(modf_slp_era_interim, 'g^')
    plt.plot(modf_slp_era_40, 'bs')
    plt.plot(modf_slp, 'r--')
    plt.legend(['Msl era interim', 'Msl era40', 'Msl combined'])
    plt.show()

    # Save results at output
    output_csv_file_name = 'guadalete_estuary_sea_level_pressure.csv'
    output_csv_path = os.path.join('output', 'csv')
    output_modf_file_name = 'guadalete_estuary_sea_level_pressure.modf'
    output_modf_path = os.path.join('output', 'modf')

    tests.save_to_csv(modf_slp, output_csv_file_name, output_csv_path)
    output_modf = os.path.join(output_modf_path, output_modf_file_name)
    modf_slp.to_file(output_modf)


def test_sea_level_pressure_pg():
    # Read the first file (era interim until 1979)
    input_file_name = 'data.nc'
    input_path = os.path.join(tests.full_data_path, 'locations', 'pg', 'sea_level_pressure', 'ERA_granada')
    metadata = {'source': 'ERA(ECMWF)'}
    modf_slp_era_interim = era.sea_level_pressure(input_file_name, metadata, input_path)

    # Read the second file (era 40 from 1979)
    input_path = os.path.join(tests.full_data_path, 'locations', 'pg', 'sea_level_pressure', 'ERA40_granada')
    metadata = {'source': 'ERA(ECMWF)'}
    modf_slp_era_40 = era.sea_level_pressure(input_file_name, metadata, input_path)

    # Combination of both files
    modf_slp = modf_slp_era_interim.combine_first(modf_slp_era_40)

    # Plot results
    plt.figure()
    plt.plot(modf_slp_era_interim, 'g^')
    plt.plot(modf_slp_era_40, 'bs')
    plt.plot(modf_slp, 'r--')
    plt.legend(['Msl era interim', 'Msl era40', 'Msl combined'])
    plt.show()

    # Save results at output
    output_csv_file_name = 'granada_beach_sea_level_pressure.csv'
    output_csv_path = os.path.join('output', 'csv')
    output_modf_file_name = 'granada_beach_sea_level_pressure.modf'
    output_modf_path = os.path.join('output', 'modf')

    tests.save_to_csv(modf_slp, output_csv_file_name, output_csv_path)
    output_modf = os.path.join(output_modf_path, output_modf_file_name)
    modf_slp.to_file(output_modf)


def test_sea_level_pressure_em():
    # Read the first file (era interim until 1979)
    input_file_name = 'data.nc'
    input_path = os.path.join(tests.full_data_path, 'locations', 'em', 'sea_level_pressure', 'ERA_portugal')
    metadata = {'source': 'ERA(ECMWF)'}
    modf_slp_era_interim = era.sea_level_pressure(input_file_name, metadata, input_path)

    # Read the second file (era 40 from 1979)
    input_path = os.path.join(tests.full_data_path, 'locations', 'em', 'sea_level_pressure', 'ERA40_portugal')
    metadata = {'source': 'ERA(ECMWF)'}
    modf_slp_era_40 = era.sea_level_pressure(input_file_name, metadata, input_path)

    # Combination of both files
    modf_slp = modf_slp_era_interim.combine_first(modf_slp_era_40)

    # Plot results
    plt.figure()
    plt.plot(modf_slp_era_interim, 'g^')
    plt.plot(modf_slp_era_40, 'bs')
    plt.plot(modf_slp, 'r--')
    plt.legend(['Msl era interim', 'Msl era40', 'Msl combined'])
    plt.show()

    # Save results at output
    output_csv_file_name = 'mondego_estuary_sea_level_pressure.csv'
    output_csv_path = os.path.join('output', 'csv')
    output_modf_file_name = 'mondego_estuary_sea_level_pressure.modf'
    output_modf_path = os.path.join('output', 'modf')

    tests.save_to_csv(modf_slp, output_csv_file_name, output_csv_path)
    output_modf = os.path.join(output_modf_path, output_modf_file_name)
    modf_slp.to_file(output_modf)


def test_sea_level_pressure_cc():
    # Read the first file (era interim until 1979)
    input_file_name = 'data.nc'
    input_path = os.path.join(tests.full_data_path, 'locations', 'cc', 'sea_level_pressure', 'ERA_cancun')
    metadata = {'source': 'ERA(ECMWF)'}
    modf_slp_era_interim = era.sea_level_pressure(input_file_name, metadata, input_path)

    # Read the second file (era 40 from 1979)
    input_path = os.path.join(tests.full_data_path, 'locations', 'cc', 'sea_level_pressure', 'ERA40_cancun')
    metadata = {'source': 'ERA(ECMWF)'}
    modf_slp_era_40 = era.sea_level_pressure(input_file_name, metadata, input_path)

    # Combination of both files
    modf_slp = modf_slp_era_interim.combine_first(modf_slp_era_40)

    # Plot results
    plt.figure()
    plt.plot(modf_slp_era_interim, 'g^')
    plt.plot(modf_slp_era_40, 'bs')
    plt.plot(modf_slp, 'r--')
    plt.legend(['Msl era interim', 'Msl era40', 'Msl combined'])
    plt.show()

    # Save results at output
    output_csv_file_name = 'cancun_sea_level_pressure.csv'
    output_csv_path = os.path.join('output', 'csv')
    output_modf_file_name = 'cancun_sea_level_pressure.modf'
    output_modf_path = os.path.join('output', 'modf')

    tests.save_to_csv(modf_slp, output_csv_file_name, output_csv_path)
    output_modf = os.path.join(output_modf_path, output_modf_file_name)
    modf_slp.to_file(output_modf)


def test_sea_level_pressure_jl():
    # Read the first file (era interim until 1979)
    input_file_name = 'data.nc'
    input_path = os.path.join(tests.full_data_path, 'locations', 'jl', 'sea_level_pressure', 'ERA_juan_lacaze')
    metadata = {'source': 'ERA(ECMWF)'}
    modf_slp_era_interim = era.sea_level_pressure(input_file_name, metadata, input_path)

    # Read the second file (era 40 from 1979)
    input_path = os.path.join(tests.full_data_path, 'locations', 'jl', 'sea_level_pressure', 'ERA40_juan_lacaze')
    metadata = {'source': 'ERA(ECMWF)'}
    modf_slp_era_40 = era.sea_level_pressure(input_file_name, metadata, input_path)

    # Combination of both files
    modf_slp = modf_slp_era_interim.combine_first(modf_slp_era_40)

    # Plot results
    plt.figure()
    plt.plot(modf_slp_era_interim, 'g^')
    plt.plot(modf_slp_era_40, 'bs')
    plt.plot(modf_slp, 'r--')
    plt.legend(['Msl era interim', 'Msl era40', 'Msl combined'])
    plt.show()

    # Save results at output
    output_csv_file_name = 'juan_lacaze_sea_level_pressure.csv'
    output_csv_path = os.path.join('output', 'csv')
    output_modf_file_name = 'juan_lacaze_sea_level_pressure.modf'
    output_modf_path = os.path.join('output', 'modf')

    tests.save_to_csv(modf_slp, output_csv_file_name, output_csv_path)
    output_modf = os.path.join(output_modf_path, output_modf_file_name)
    modf_slp.to_file(output_modf)


def test_sea_level_pressure_vp():
    # Read the first file (era interim until 1979)
    input_file_name = 'data.nc'
    input_path = os.path.join(tests.full_data_path, 'locations', 'vp', 'sea_level_pressure', 'ERA_valparaiso')
    metadata = {'source': 'ERA(ECMWF)'}
    modf_slp_era_interim = era.sea_level_pressure(input_file_name, metadata, input_path)

    # Read the second file (era 40 from 1979)
    input_path = os.path.join(tests.full_data_path, 'locations', 'vp', 'sea_level_pressure', 'ERA40_valparaiso')
    metadata = {'source': 'ERA(ECMWF)'}
    modf_slp_era_40 = era.sea_level_pressure(input_file_name, metadata, input_path)

    # Combination of both files
    modf_slp = modf_slp_era_interim.combine_first(modf_slp_era_40)

    # Plot results
    plt.figure()
    plt.plot(modf_slp_era_interim, 'g^')
    plt.plot(modf_slp_era_40, 'bs')
    plt.plot(modf_slp, 'r--')
    plt.legend(['Msl era interim', 'Msl era40', 'Msl combined'])
    plt.show()

    # Save results at output
    output_csv_file_name = 'gran_valparaiso_sea_level_pressure.csv'
    output_csv_path = os.path.join('output', 'csv')
    output_modf_file_name = 'gran_valparaiso_sea_level_pressure.modf'
    output_modf_path = os.path.join('output', 'modf')

    tests.save_to_csv(modf_slp, output_csv_file_name, output_csv_path)
    output_modf = os.path.join(output_modf_path, output_modf_file_name)
    modf_slp.to_file(output_modf)