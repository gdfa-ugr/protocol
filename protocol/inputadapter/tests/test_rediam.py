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
from inputadapter import rediam, saih, tests


def test_river_discharge():
    file_name = 'caudal_desembalsado'
    dams = ['270_BORNOS', '273_GUADALCACIN']
    path = os.path.join(tests.full_data_path, 'rediam')
    metadata = {'id': '270_BORNOS_273_GUADALCACIN',
                'latitude': None,
                'longitude': None}
    modf_river_disch = rediam.river_discharge(file_name, metadata, dams, path)

    assert modf_river_disch.driver == 'river_discharge'


def test_all_drivers():
    file_name = 'caudal_desembalsado'
    dams = ['270_BORNOS', '273_GUADALCACIN']
    path = os.path.join(tests.full_data_path, 'rediam')
    metadata = {'id': '270_BORNOS_273_GUADALCACIN',
                'latitude': None,
                'longitude': None}
    modfs = rediam.all_drivers(file_name, metadata, dams, path)


def test_river_discharge_eg():
    input_file_name = 'caudal_desembalsado'
    dams = ['270_BORNOS', '273_GUADALCACIN']
    input_path = os.path.join(tests.full_data_path, 'locations', 'eg', 'river_flow', 'rediam')
    output_file_name = 'guadalete_estuary_river_discharge.csv'
    output_path = os.path.join('output', 'csv')
    metadata = {'source': 'rediam',
                'id': '270_BORNOS_273_GUADALCACIN',
                'latitude': 36.693,
                'longitude': -5.858}

    # Adapt driver
    modf_river_disch_rediam = rediam.river_discharge(input_file_name, metadata, dams, input_path)

    # Read the data from de saih from 1972
    input_file_name = 'river_flow.txt'
    input_path = os.path.join(tests.full_data_path, 'locations', 'eg', 'river_flow', 'saih')
    modf_river_disch_saih = saih.river_discharge(input_file_name, metadata, input_path)

    # Combine
    modf_river_disch = modf_river_disch_saih.combine_first(modf_river_disch_rediam)

    # Plot results
    fig = plt.figure()
    ax = fig.gca()
    ax.plot(modf_river_disch_saih)
    plt.hold
    ax.plot(modf_river_disch_rediam, '.k', markersize=2)
    ax.plot(modf_river_disch, 'r')
    plt.show()

    # Save results at output
    tests.save_to_csv(modf_river_disch, output_file_name, output_path)
