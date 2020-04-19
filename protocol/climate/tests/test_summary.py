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

from climate import summary, tests
from climate.util import plot
from climate.stats import empirical_distributions
from input import saih, tidal_model_driver
from metoceandataframe.metoceandataframe import MetOceanDF


def test_get_summary():
    data_simar = tests.read_sample_simar()

    data_summary = summary.get_summary(data_simar)

    assert data_summary.loc['max', 'Hm0'] == 2.4


def test_plot_trends():
    data_simar = tests.read_sample_simar()

    summary.plot_series(data_simar, data_column='Hm0', title='Trends', var_name='$H_s$', var_unit='m',
                        fig_filename=tests.get_img_path('trends.png'), show_trends=True)


def test_plot_series():
    data_simar = tests.read_sample_simar()

    summary.plot_series(data_simar['Hm0'], title='Wave', var_name='$H_s$', var_unit='m',
                        fig_filename=tests.get_img_path('series.png'))
    summary.plot_series(data_simar, data_column='Hm0', title='Wave', var_name='$H_s$', var_unit='m')


def test_plot_series_circular():
    data_simar = tests.read_sample_simar()

    summary.plot_series(data_simar, data_column='DirM', title='Series Circular', var_name='$Dir_W$', var_unit='º',
                        fig_filename=tests.get_img_path('series_circular.png'), circular=True)


def test_plot_rose():
    data_simar = tests.read_sample_simar()

    summary.plot_rose(data_simar, data_column='Hm0', dir_column='DirM', title='Wave Rose', var_name='$H_s$',
                      var_unit='m', fig_filename=tests.get_img_path('rose.png'))


def test_plot_rose_granada():
    # Data
    modf = os.path.join(tests.current_path, '..', '..', 'inputadapter',  'tests', 'output', 'modf',
                        'granada_beach_wave.modf')
    modf = MetOceanDF.read_file(modf)

    summary.plot_rose(modf, data_column='Hm0', dir_column='DirM', title='Wave Rose', var_name='$H_s$',
                      var_unit='m', fig_filename=tests.get_img_path('rose_granada.png'))


def test_plot_rose_granada():
    # Data
    modf = os.path.join(tests.current_path, '..', '..', 'inputadapter',  'tests', 'output', 'modf',
                        'guadalete_estuary_wave.modf')
    modf = MetOceanDF.read_file(modf)

    summary.plot_rose(modf, data_column='Hm0', dir_column='DirM', title='Wave Rose', var_name='$H_s$',
                      var_unit='m', fig_filename=tests.get_img_path('rose_cadiz.png'))


def test_plot_histogram():
    data_simar = tests.read_full_simar()

    summary.plot_histogram(data_simar['Hm0'], title='Histogram', var_name='$H_s$',
                           var_unit='m', fig_filename='', kernel=True)


def test_plot_histogram_circular():
    data_simar = tests.read_sample_simar()

    summary.plot_histogram(data_simar['DirM'], title='Histogram', var_name='$Dir_W$',
                           var_unit='º', circular=True,
                           fig_filename=tests.get_img_path('histogram_circular.png'))


def test_plot_scatter():
    data_simar = tests.read_sample_simar()

    summary.plot_scatter(data_simar, 'Hm0', 'VelV', title='Scatter', x_var_name='$H_s$', x_var_unit='m',
                         y_var_name='$Vel_v$', y_var_unit='m/s',
                         fig_filename=tests.get_img_path('scatter.png'))


def test_plot_scatter_circular():
    data_simar = tests.read_sample_simar()

    summary.plot_scatter(data_simar, 'DirM', 'Hm0', title='Scatter Circular', x_var_name='$Dir_W$',
                         x_var_unit='º', y_var_name='$H_s$', y_var_unit='m', circular='x',
                         fig_filename=tests.get_img_path('scatter_circular.png'))


def test_plot_variability():
    data_simar = tests.read_sample_simar('SIMAR_1052046', os.path.join(tests.full_data_path, 'simar'))

    summary.plot_variability(data_simar['Hm0'], 'month', title='Monthly variability', var_name='$H_s$', var_unit='m',
                             fig_filename=tests.get_img_path('monthly_variability.png'))
    summary.plot_variability(data_simar.loc['2000':, 'Hm0'], 'year', title='Yearly variability', var_name='$H_s$',
                             var_unit='m', fig_filename=tests.get_img_path('yearly_variability.png'))
    summary.plot_variability(data_simar['Hm0'], 'dayofyear', title='Day of year variability', var_name='$H_s$',
                             var_unit='m', fig_filename=tests.get_img_path('dayofyear_variability.png'))


def test_plot_variability_circular():
    data_simar = tests.read_sample_simar('SIMAR_1052046', os.path.join(tests.full_data_path, 'simar'))

    summary.plot_variability(data_simar['DirM'], 'month', title='Monthly variability circular', var_name='$Dir_W$',
                             var_unit='º', circular=True,
                             fig_filename=tests.get_img_path('monthly_variability_circular.png'))


def test_plot_anual_variability():
    data_simar = tests.read_sample_simar('SIMAR_1052046', os.path.join(tests.full_data_path, 'simar'))

    summary.plot_anual_variability(data_simar.loc['2000':'2017', 'Hm0'], 'mean',
                                   fig_filename=tests.get_img_path('dayofyear_variability.png'))


def test_plot_full_simar_series():
    simar_name = 'SIMAR_1052046'
    data_path = os.path.join(tests.full_data_path, 'simar')

    # Read SIMAR
    data_simar = tests.read_sample_simar(data_file=simar_name, data_path=data_path)

    # Plot series
    summary.plot_series(data_simar, title='Plot series', var_name='Hm0', var_unit='m',
                        fig_filename=tests.get_img_path('time_series_full_hs.png'), data_column='Hm0', 
						circular=False, show_trends=False, label='observations')


def test_wind_rose():
    simar_name = 'SIMAR_1052046'
    data_path = os.path.join(tests.full_data_path, 'simar')

    # Read SIMAR
    data_simar = tests.read_sample_simar(data_file=simar_name, data_path=data_path)

    summary.plot_rose(data_simar, data_column='VelV', dir_column='DirV', title='Wind Rose', var_name='$W_v$',
                      var_unit='m', fig_filename=tests.get_img_path('wind_rose.png'))


def test_astronomical_tide_histogram():
    name = 'data.out'
    data_path = os.path.join(tests.full_data_path, 'tidal_model_driver', name)
    data = tidal_model_driver.read_time_series_file(data_path)

    # Plot series
    summary.plot_histogram(data, title='Histogram', var_name='$\eta_{AT}$',
                           var_unit='$m$', fig_filename=tests.get_img_path('histogram.png'))


def test_river_discharge_ecdf():
    name = 'river_flow.txt'
    data_path = os.path.join(tests.full_data_path, 'saih', name)

    # Read Saih
    data = saih.read_file(data_path)

    # Empirical cdf
    cumulative = True
    data = empirical_distributions.kde_sm(data, cumulative=cumulative)

    # Plot
    plt.figure()
    plt.show()
    ax = plt.axes()
    ax.plot(data.index, data)
    plot.plot_title(ax, 'Cummulative density function')
    ax.set_xlabel('$Q (m^3 \, s^{-1})$')
    ax.set_ylabel('CDF')
    plot.save_figure('CDF_river_discharge')


def test_scatter_simar():
    simar_name = 'SIMAR_1052046'
    data_path = os.path.join(tests.full_data_path, 'simar')

    # Read SIMAR
    data_simar = tests.read_sample_simar(data_file=simar_name, data_path=data_path)

    summary.plot_scatter(data_simar, 'DirM', 'Hm0', title='Scatter Circular', x_var_name='$Dir_W$',
                         x_var_unit='º', y_var_name='$H_s$', y_var_unit='m', circular='x',
                         fig_filename=tests.get_img_path('scatter_circular.png'))


def test_annual_variability_river_discharge():
    name = 'river_flow.txt'
    data_path = os.path.join(tests.full_data_path, 'saih', name)

    # Read Saih
    data = saih.read_file(data_path)

    summary.plot_variability(data[4], 'dayofyear', title='Day of year variability', var_name='$Q$',
                             var_unit='$m^3 \, s^{-1}$', fig_filename=tests.get_img_path('dayofyear_variability.png'))
