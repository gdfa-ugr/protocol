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
from inputadapter import snirh, tests


def test_river_discharge():
    file_name = 'river_flow.csv'
    metadata = {'id': 'AC_Ponte_Mocate_APC_Azude_Ponte_Coimbra',
                'latitude': None,
                'longitude': None}
    path = os.path.join(tests.sample_data_path, 'snirh')

    modf_river_disch = snirh.river_discharge(file_name, metadata, path)

    assert modf_river_disch.id == 'AC_Ponte_Mocate_APC_Azude_Ponte_Coimbra'


def test_all_drivers():
    file_name = 'river_flow.csv'
    metadata = {'id': 'AC_Ponte_Mocate_APC_Azude_Ponte_Coimbra',
                'latitude': None,
                'longitude': None}
    path = os.path.join(tests.sample_data_path, 'snirh')

    modfs = snirh.all_drivers(file_name, metadata, path)


def test_river_discharge_em():
    input_file_name = 'descarga_fluvial.csv'
    input_path = os.path.join(tests.full_data_path, 'locations', 'em', 'river_flow')
    output_csv_file_name = 'mondego_estuary_river_discharge.csv'
    output_csv_path = os.path.join('output', 'csv')
    output_modf_file_name = 'mondego_estuary_river_discharge.modf'
    output_modf_path = os.path.join('output', 'modf')
    metadata = {'source': 'SNIRH',
                'id':  'AC_Ponte_Mocate_APC_Azude_Ponte_Coimbra',
                'latitude': 40.216,
                'longitude': -8.440}

    # Adapt driver
    modf_river_disch = snirh.river_discharge(input_file_name, metadata, input_path)

    # Plot results
    fig = plt.figure()
    ax = fig.gca()
    ax.plot(modf_river_disch)
    plt.show()

    # Save results at output
    tests.save_to_csv(modf_river_disch, output_csv_file_name, output_csv_path)
    output_modf = os.path.join(output_modf_path, output_modf_file_name)
    modf_river_disch.to_file(output_modf)
