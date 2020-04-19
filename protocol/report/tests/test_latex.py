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

from metoceandataframe.metoceandataframe import MetOceanDF
from preprocessing import missing_values
from report import latex, tests


def test_create_latex_document_granada_beach():
    location = 'granada_beach'
    drivers = ['wave', 'wind', 'astronomical_tide', 'sea_level_pressure']

    data = []
    # Data
    for driver in drivers:
        modf = os.path.join(tests.current_path, '..', '..', 'inputadapter',  'tests', 'output', 'modf',
                            '{}_{}.modf'.format(location, driver))
        data.append(MetOceanDF.read_file(modf))

    # Config report file
    template = os.path.join(tests.current_path, '..', 'templates', 'latex', '{}.conf'.format(location))

    latex.create_document(data, template, output_title=location)


def test_create_latex_document_cancun():
    location = 'cancun'
    drivers = ['wave', 'wind', 'astronomical_tide', 'sea_level_pressure']

    data = []
    # Data
    for driver in drivers:
        modf = os.path.join(tests.current_path, '..', '..', 'inputadapter',  'tests', 'output', 'modf',
                            '{}_{}.modf'.format(location, driver))
        data.append(MetOceanDF.read_file(modf))

    # Config report file
    template = os.path.join(tests.current_path, '..', 'templates', 'latex', '{}.conf'.format(location))

    latex.create_document(data, template, output_title=location)


#%% GUADALETE ESTUARY


def test_create_latex_document_guadalete_estuary_wave():
    # Data
    modf = os.path.join(tests.current_path, '..', '..', 'inputadapter',  'tests', 'output', 'modf',
                        'guadalete_estuary_wave.modf')
    modf = MetOceanDF.read_file(modf)

    # Config report file
    template = os.path.join(tests.current_path, '..', 'templates', 'latex', 'guadalete_estuary_wave.conf')

    latex.create_document(modf, template, output_title='guadalete_estuary_wave')


def test_create_latex_document_guadalete_estuary_wind():
    # Data
    modf = os.path.join(tests.current_path, '..', '..', 'inputadapter',  'tests', 'output', 'modf',
                        'guadalete_estuary_wind.modf')
    modf = MetOceanDF.read_file(modf)

    # Config report file
    template = os.path.join(tests.current_path, '..', 'templates', 'latex', 'guadalete_estuary_wind.conf')

    latex.create_document(modf, template, output_title='guadalete_estuary_wind')


def test_create_latex_document_guadalete_estuary_river_discharge():
    # Data
    modf = os.path.join(tests.current_path, '..', '..', 'inputadapter',  'tests', 'output', 'modf',
                        'guadalete_estuary_river_discharge.modf')
    modf = MetOceanDF.read_file(modf)

    # Config report file
    template = os.path.join(tests.current_path, '..', 'templates', 'latex', 'guadalete_estuary_river_discharge.conf')

    latex.create_document(modf, template, output_title='guadalete_estuary_river_discharge')


def test_create_latex_document_guadalete_estuary_astronomical_tide():
    # Data
    modf = os.path.join(tests.current_path, '..', '..', 'inputadapter',  'tests', 'output', 'modf',
                        'guadalete_estuary_astronomical_tide.modf')
    modf = MetOceanDF.read_file(modf)

    # Config report file
    template = os.path.join(tests.current_path, '..', 'templates', 'latex', 'guadalete_estuary_astronomical_tide.conf')

    latex.create_document(modf, template, output_title='guadalete_estuary_astronomical_tide')


def test_create_latex_document_guadalete_estuary_sea_level_pressure():
    # Data
    modf = os.path.join(tests.current_path, '..', '..', 'inputadapter',  'tests', 'output', 'modf',
                        'guadalete_estuary_sea_level_pressure.modf')
    modf = MetOceanDF.read_file(modf)

    # Config report file
    template = os.path.join(tests.current_path, '..', 'templates', 'latex', 'guadalete_estuary_sea_level_pressure.conf')

    latex.create_document(modf, template, output_title='guadalete_estuary_sea_level_pressure')

#%% GRANADA BEACH


def test_create_latex_document_granada_beach_wave():
    # Data
    modf = os.path.join(tests.current_path, '..', '..', 'inputadapter',  'tests', 'output', 'modf',
                        'granada_beach_wave.modf')
    modf = MetOceanDF.read_file(modf)

    # Config report file
    template = os.path.join(tests.current_path, '..', 'templates', 'latex', 'granada_beach_wave.conf')

    latex.create_document(modf, template, output_title='granada_beach_wave')


def test_create_latex_document_granada_beach_wind():
    # Data
    modf = os.path.join(tests.current_path, '..', '..', 'inputadapter',  'tests', 'output', 'modf',
                        'granada_beach_wind.modf')
    modf = MetOceanDF.read_file(modf)

    # Config report file
    template = os.path.join(tests.current_path, '..', 'templates', 'latex', 'granada_beach_wind.conf')

    latex.create_document(modf, template, output_title='granada_beach_wind')


def test_create_latex_document_granada_beach_astronomical_tide():
    # Data
    modf = os.path.join(tests.current_path, '..', '..', 'inputadapter',  'tests', 'output', 'modf',
                        'granada_beach_astronomical_tide.modf')
    modf = MetOceanDF.read_file(modf)

    # Config report file
    template = os.path.join(tests.current_path, '..', 'templates', 'latex', 'granada_beach_astronomical_tide.conf')

    latex.create_document(modf, template, output_title='granada_beach_astronomical_tide')


def test_create_latex_document_granada_beach_sea_level_pressure():
    # Data
    modf = os.path.join(tests.current_path, '..', '..', 'inputadapter',  'tests', 'output', 'modf',
                        'granada_beach_sea_level_pressure.modf')
    modf = MetOceanDF.read_file(modf)

    # Config report file
    template = os.path.join(tests.current_path, '..', 'templates', 'latex', 'granada_beach_sea_level_pressure.conf')

    latex.create_document(modf, template, output_title='ggranada_beach_sea_level_pressure')


#%% MONDEGO ESTUARY


def test_create_latex_document_mondego_estuary_wave():
    # Data
    modf = os.path.join(tests.current_path, '..', '..', 'inputadapter',  'tests', 'output', 'modf',
                        'mondego_estuary_wave_I.modf')
    modf = MetOceanDF.read_file(modf)

    # Config report file
    template = os.path.join(tests.current_path, '..', 'templates', 'latex', 'mondego_estuary_wave.conf')

    latex.create_document(modf, template, output_title='mondego_estuary_wave_I')


def test_create_latex_document_mondego_estuary_wind():
    # Data
    modf = os.path.join(tests.current_path, '..', '..', 'inputadapter',  'tests', 'output', 'modf',
                        'mondego_estuary_wind_I.modf')
    modf = MetOceanDF.read_file(modf)

    # Config report file
    template = os.path.join(tests.current_path, '..', 'templates', 'latex', 'mondego_estuary_wind.conf')

    latex.create_document(modf, template, output_title='mondego_estuary_wind_I')


def test_create_latex_document_mondego_estuary_river_discharge():
    # Data
    modf = os.path.join(tests.current_path, '..', '..', 'inputadapter',  'tests', 'output', 'modf',
                        'mondego_estuary_river_discharge.modf')
    modf = MetOceanDF.read_file(modf)

    # Config report file
    template = os.path.join(tests.current_path, '..', 'templates', 'latex', 'mondego_estuary_river_discharge.conf')

    latex.create_document(modf, template, output_title='mondego_estuary_river_discharge')


def test_create_latex_document_mondego_estuary_astronomical_tide():
    # Data
    modf = os.path.join(tests.current_path, '..', '..', 'inputadapter',  'tests', 'output', 'modf',
                        'mondego_estuary_astronomical_tide.modf')
    modf = MetOceanDF.read_file(modf)

    # Config report file
    template = os.path.join(tests.current_path, '..', 'templates', 'latex', 'mondego_estuary_astronomical_tide.conf')

    latex.create_document(modf, template, output_title='mondego_estuary_astronomical_tide')


def test_create_latex_document_mondego_estuary_sea_level_pressure():
    # Data
    modf = os.path.join(tests.current_path, '..', '..', 'inputadapter',  'tests', 'output', 'modf',
                        'mondego_estuary_sea_level_pressure.modf')
    modf = MetOceanDF.read_file(modf)

    # Config report file
    template = os.path.join(tests.current_path, '..', 'templates', 'latex', 'mondego_estuary_sea_level_pressure.conf')

    latex.create_document(modf, template, output_title='mondego_estuary_sea_level_pressure')


#%% Cancun
def test_create_latex_document_cancun_wave():
    # Data
    modf = os.path.join(tests.current_path, '..', '..', 'inputadapter',  'tests', 'output', 'modf',
                        'cancun_wave_I.modf')
    modf = MetOceanDF.read_file(modf)
    # Remove duplicates
    modf_unique = missing_values.erase_duplicated_time_indices(modf, 'first')

    # Config report file
    template = os.path.join(tests.current_path, '..', 'templates', 'latex', 'cancun_wave.conf')

    latex.create_document(modf_unique, template, output_title='cancun_wave')


def test_create_latex_document_cancun_wind():
    # Data
    modf = os.path.join(tests.current_path, '..', '..', 'inputadapter',  'tests', 'output', 'modf',
                        'cancun_wind_I.modf')
    modf = MetOceanDF.read_file(modf)
    # Remove duplicates
    modf_unique = missing_values.erase_duplicated_time_indices(modf, 'first')

    # Config report file
    template = os.path.join(tests.current_path, '..', 'templates', 'latex', 'cancun_wind.conf')

    latex.create_document(modf_unique, template, output_title='cancun_wind')


def test_create_latex_document_cancun_astronomical_tide():
    # Data
    modf = os.path.join(tests.current_path, '..', '..', 'inputadapter',  'tests', 'output', 'modf',
                        'cancun_astronomical_tide.modf')
    modf = MetOceanDF.read_file(modf)

    # Config report file
    template = os.path.join(tests.current_path, '..', 'templates', 'latex', 'cancun_astronomical_tide.conf')

    latex.create_document(modf, template, output_title='cancun_astronomical_tide')


def test_create_latex_document_cancun_sea_level_pressure():
    # Data
    modf = os.path.join(tests.current_path, '..', '..', 'inputadapter',  'tests', 'output', 'modf',
                        'cancun_sea_level_pressure.modf')
    modf = MetOceanDF.read_file(modf)

    # Config report file
    template = os.path.join(tests.current_path, '..', 'templates', 'latex', 'cancun_sea_level_pressure.conf')

    latex.create_document(modf, template, output_title='cancun_sea_level_pressure')


#%% Juan Lacaze
def test_create_latex_document_juan_lacaze_wave():
    # Data
    modf = os.path.join(tests.current_path, '..', '..', 'inputadapter',  'tests', 'output', 'modf',
                        'juan_lacaze_wave.modf')
    modf = MetOceanDF.read_file(modf)
    # Remove duplicates
    modf_unique = missing_values.erase_duplicated_time_indices(modf, 'first')

    # Config report file
    template = os.path.join(tests.current_path, '..', 'templates', 'latex', 'juan_lacaze_wave.conf')

    latex.create_document(modf_unique, template, output_title='juan_lacaze_wave')


def test_create_latex_document_juan_lacaze_astronomical_tide():
    # Data
    modf = os.path.join(tests.current_path, '..', '..', 'inputadapter',  'tests', 'output', 'modf',
                        'juan_lacaze_astronomical_tide.modf')
    modf = MetOceanDF.read_file(modf)

    # Config report file
    template = os.path.join(tests.current_path, '..', 'templates', 'latex', 'juan_lacaze_astronomical_tide.conf')

    latex.create_document(modf, template, output_title='juan_lacaze_astronomical_tide')


def test_create_latex_document_juan_lacaze_sea_level_pressure():
    # Data
    modf = os.path.join(tests.current_path, '..', '..', 'inputadapter',  'tests', 'output', 'modf',
                        'juan_lacaze_sea_level_pressure.modf')
    modf = MetOceanDF.read_file(modf)

    # Config report file
    template = os.path.join(tests.current_path, '..', 'templates', 'latex', 'juan_lacaze_sea_level_pressure.conf')

    latex.create_document(modf, template, output_title='juan_lacaze_sea_level_pressure')


#%% Valparaiso
def test_create_latex_document_gran_valparaiso_wave():
    # Data
    modf = os.path.join(tests.current_path, '..', '..', 'inputadapter',  'tests', 'output', 'modf',
                        'gran_valparaiso_wave.modf')
    modf = MetOceanDF.read_file(modf)
    # Remove duplicates
    modf_unique = missing_values.erase_duplicated_time_indices(modf, 'first')

    # Config report file
    template = os.path.join(tests.current_path, '..', 'templates', 'latex', 'gran_valparaiso_wave.conf')

    latex.create_document(modf_unique, template, output_title='gran_valparaiso_wave')


def test_create_latex_document_gran_valparaiso_astronomical_tide():
    # Data
    modf = os.path.join(tests.current_path, '..', '..', 'inputadapter',  'tests', 'output', 'modf',
                        'gran_valparaiso_astronomical_tide.modf')
    modf = MetOceanDF.read_file(modf)

    # Config report file
    template = os.path.join(tests.current_path, '..', 'templates', 'latex', 'gran_valparaiso_astronomical_tide.conf')

    latex.create_document(modf, template, output_title='gran_valparaiso_astronomical_tide')


def test_create_latex_document_gran_valparaiso_sea_level_pressure():
    # Data
    modf = os.path.join(tests.current_path, '..', '..', 'inputadapter',  'tests', 'output', 'modf',
                        'gran_valparaiso_sea_level_pressure.modf')
    modf = MetOceanDF.read_file(modf)

    # Config report file
    template = os.path.join(tests.current_path, '..', 'templates', 'latex', 'gran_valparaiso_sea_level_pressure.conf')

    latex.create_document(modf, template, output_title='gran_valparaiso_sea_level_pressure')
