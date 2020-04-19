#!/usr/bin/env python
# coding: utf-8
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import numpy as np

try:
    # noinspection PyUnresolvedReferences, PyCompatibility
    from builtins import *  # noqa
except ImportError:
    pass

import os

import pandas as pd
from climate import tests, analysis, read
from climate.stats import extremal, empirical_distributions, fitting
from metoceandataframe.metoceandataframe import MetOceanDF
import matplotlib.pyplot as plt


def test_annual_maxima_calculation():
    # Read preprocessed SIMAR
    data = pd.read_msgpack(os.path.join(tests.full_data_path, 'intermediate_files', 'full_simar_preprocessed.msg'))
    hs = data.loc[:, 'Hm0']
    # Calculation of the annual maxima sample
    annual_maxima = extremal.annual_maxima_calculation(hs)
    # Plot
    file_name = os.path.join(tests.current_path, 'output', 'img', 'Annual_maxima_peaks')
    extremal.peaks_plot(data, annual_maxima, data_column='Hm0', title='Annual Maxima', var_name='Hm0', var_unit='m',
                        fig_filename=file_name, circular=False, data_label='Hm0', peaks_label='Annual Maxima')

    assert annual_maxima[2005] == 4.3


def test_extreme_events():
    values = [2.3, 0.5, 1, 1.9, 3, 3.9, 4.8, 1.6, 1.1, 2.5, 1.5, 1.7, 2.1, 2.8, 3.1, 1.7, 0.8, 2.1]

    data = pd.Series(values, index=pd.date_range('1980-01-01 00:00:00', periods=len(values), freq='1H'))
    threshold = 2
    cycles, calm_periods = extremal.extreme_events(data, threshold, pd.Timedelta('1H'), pd.Timedelta('1H'),
                                                   truncate=True)

    maximum = extremal.events_max(cycles)

    # frequency = pd.Series(data.index).diff().min()
    duration = extremal.events_duration(cycles)

    magnitude = extremal.events_magnitude(cycles, threshold)

    # noinspection PyTypeChecker
    print(len(cycles) == len(calm_periods))


def test_extreme_events_full_simar_with_interpolation():
    data_file = 'SIMAR_1052046'
    full_data_path = os.path.join('..', '..', '..', '..', '..', 'data', 'simar')
    data_simar, code = read.simar(data_file, path=full_data_path)

    var_name = 'Hm0'
    threshold = np.percentile(data_simar[var_name], 95)
    minimum_interarrival_time = pd.Timedelta('3 days')
    minimum_cycle_length = pd.Timedelta('3 hours')

    cycles, calm_periods = extremal.extreme_events(data_simar, var_name, threshold, minimum_interarrival_time,
                                                   minimum_cycle_length, interpolation=True)

    test = extremal.events_boundaries(cycles)
    maximum = extremal.events_max(cycles)

    # frequency = pd.Series(data.index).diff().min()
    duration = extremal.events_duration(cycles)

    magnitude = extremal.events_magnitude(cycles, threshold)

    # noinspection PyTypeChecker
    print(len(cycles) == len(calm_periods))


def test_extremal_annual_maxima():
    data_column = 'hs'
    cumulative = True

    # Read preprocessed SIMAR
    data = pd.read_msgpack(os.path.join(tests.full_data_path, 'intermediate_files', 'full_simar_preprocessed.msg'))
    hs = data.loc[:, 'Hm0']
    # Calculation of the annual maxima sample
    annual_maxima = extremal.annual_maxima_calculation(hs)
    # Conversion to DataFrame
    annual_maxima.name = data_column
    annual_maxima_df = annual_maxima.to_frame()

    # Empirical CDF using statsmodel
    annual_maxima_ecdf = empirical_distributions.ecdf_sm(annual_maxima)
    x_annual_maxima_ecdf = annual_maxima_ecdf.index.values
    y_annual_maxima_ecdf = annual_maxima_ecdf.values

    # Empirical CDF Kdensity
    annual_maxima_ecdf_kde = empirical_distributions.kde_sm(annual_maxima, kernel='gau', bw='scott', gridsize=100,
                                                            cut=3, clip=(-np.inf, np.inf), cumulative=True)

    x_annual_maxima_ecdf_kde = annual_maxima_ecdf_kde.index.values
    y_annual_maxima_ecdf_kde = annual_maxima_ecdf_kde.values

    # GEV fit
    x_gev, y_gev, y_gev_return_period, _ = analysis.extremal_annual_maxima(annual_maxima_df, data_column,
                                                                           cumulative=cumulative)

    # Plot kdensity and GEV fit
    fitting.plot_fit_kde(x_gev, y_gev, annual_maxima_df, data_column, title='GEV fitting', var_name='$H_s$',
                         var_unit='m', kde=True, hist=True, cumulative=cumulative)

    # Plot ECDF, Kdensity and GEV fit
    fitting.plot_fit_kde_ecdf(x_gev, y_gev, x_annual_maxima_ecdf_kde, y_annual_maxima_ecdf_kde, x_annual_maxima_ecdf,
                              y_annual_maxima_ecdf, var_name='', var_unit='', label_fit='',
                              label_observation='', label_kde='', title='', fig_filename='')


def test_min_cycles_duration():
    # Read data
    data_file = 'SIMAR_1052046'
    full_data_path = os.path.join('..', '..', '..', '..', '..', 'data', 'simar')
    data_simar, code = read.simar(data_file, path=full_data_path)

    # Input
    threshold = np.percentile(data_simar['Hm0'], 95)
    minimum_interarrival_time = pd.Timedelta('3 days')
    minimum_cycle_length = pd.Timedelta('3 hours')

    # Cycles calculation
    cycles, calm_periods = extremal.extreme_events(data_simar, 'Hm0', threshold, minimum_interarrival_time,
                                                   minimum_cycle_length)

    # Cycles duration
    cycles_duration = extremal.events_duration(cycles)

    # Check if the min duration is equal to the set duration threshold
    min_cycles_duration = cycles_duration.min()

    # Find the cycles with duration less than the threshold
    cont = 0
    list_wrong_cycles = []
    for cycle in cycles_duration:
        if cycle == pd.Timedelta('2 hours'):
            list_wrong_cycles.append(cont)
        cont += 1

    assert min_cycles_duration == minimum_cycle_length


def test_extreme_distributions_to_peaks_values():
    # Inputs
    data_file = 'SIMAR_1052046'
    threshold_percentile = 95
    minimum_interarrival_time = pd.Timedelta('3 days')
    minimum_cycle_length = pd.Timedelta('3 hours')
    interpolation = True
    interpolation_method = 'linear'
    interpolation_freq = '1min'
    truncate = False
    extra_info = False

    # Read SIMAR
    full_data_path = os.path.join('..', '..', '..', '..', '..', 'data', 'simar')
    data_simar, code = read.simar(data_file, path=full_data_path)
    threshold = np.percentile(data_simar.loc[:, 'Hm0'], threshold_percentile)

    # Storm cycles calculation
    cycles, calm_periods = extremal.extreme_events(data_simar, 'Hm0', threshold, minimum_interarrival_time,
                                                   minimum_cycle_length, interpolation, interpolation_method,
                                                   interpolation_freq, truncate, extra_info)
    # Peaks over threshold
    peaks_over_thres = extremal.events_max(cycles)

    # Calculation of the annual maxima sample
    annual_maxima = extremal.annual_maxima_calculation(data_simar['Hm0'])

    # POT Empirical distribution
    ecdf_pot = empirical_distributions.ecdf_histogram(peaks_over_thres)
    n_peaks_year = len(peaks_over_thres) / len(data_simar['Hm0'].index.year.unique())
    ecdf_pot_rp = extremal.return_period_curve(n_peaks_year, ecdf_pot)

    # Annual Maxima Empirical distribution
    ecdf_am = empirical_distributions.ecdf_histogram(annual_maxima)
    ecdf_am_rp = extremal.return_period_curve(1, ecdf_am)

    # Fit POT to Scipy-GPD
    (param, x_gpd, y_gpd, y_gpd_rp) = extremal.extremal_distribution_fit(data=data_simar, var_name='Hm0',
                                                                         sample=peaks_over_thres,
                                                                         threshold=threshold, fit_type='gpd',
                                                                         x_min=0.90*min(peaks_over_thres),
                                                                         x_max=1.5*max(peaks_over_thres),
                                                                         n_points=1000,
                                                                         cumulative=True)

    # Fit POT to Coles-GPD
    (param, x_gpd_coles, _,
     y_gpd_rp_coles) = extremal.extremal_distribution_fit(data=data_simar, var_name='Hm0', sample=peaks_over_thres,
                                                          threshold=threshold, fit_type='coles',
                                                          x_min=0.90*min(peaks_over_thres),
                                                          x_max=1.5*max(peaks_over_thres),
                                                          n_points=1000,
                                                          cumulative=True)

    # Fit Annual Maxima to GEV
    (param, x_gev, y_gev,
     y_gev_rp) = extremal.extremal_distribution_fit(data=data_simar, var_name='Hm0', sample=annual_maxima,
                                                    threshold=None, fit_type='gev',
                                                    x_min=0.90*min(annual_maxima),
                                                    x_max=1.5*max(annual_maxima),
                                                    n_points=1000,
                                                    cumulative=True)

    # Represent results
    plt.figure()
    ax = plt.axes()
    ax.plot(x_gpd, y_gpd, label='GPD Fit')
    ax.plot(ecdf_pot.index, ecdf_pot, '.r', label='POT ECDF')
    ax.plot(ecdf_am.index, ecdf_am, '.k', label='Annual maxima ECDF')
    ax.plot(x_gev, y_gev, 'k', label='GEV Fit')
    plt.xlabel('Return Period (years)')
    plt.ylabel('Hm0 (m)')
    ax.legend()
    plt.grid()
    plt.show()

    plt.figure()
    ax = plt.axes()
    ax.semilogx(y_gpd_rp, x_gpd, label='GPD Fit')
    ax.semilogx(x_gpd_coles, y_gpd_rp_coles, 'g', label='GPD Coels Fit')
    ax.semilogx(ecdf_pot_rp, ecdf_pot_rp.index, '.r', label='POT ECDF')
    ax.semilogx(ecdf_am_rp, ecdf_am_rp.index, '.k', label='Annual maxima ECDF')
    ax.semilogx(y_gev_rp, x_gev, 'k', label='GEV Fit')
    plt.xlim(0, 500)
    plt.xlabel('Return Period (years)')
    plt.ylabel('Hm0 (m)')
    ax.legend(['GPD Fit', 'POT ECDF'])
    plt.grid()
    plt.show()


def test_poisson_pareto_fit_to_pot_and_gev_fit_to_annual_maxima():
    # Inputs
    data_file = 'SIMAR_1052046'
    threshold_percentile = 95
    minimum_interarrival_time = pd.Timedelta('3 days')
    minimum_cycle_length = pd.Timedelta('3 hours')
    interpolation = True
    interpolation_method = 'linear'
    interpolation_freq = '1min'
    truncate = False
    extra_info = False

    # Read SIMAR
    full_data_path = os.path.join('..', '..', '..', '..', '..', 'data', 'simar')
    data_simar, code = read.simar(data_file, path=full_data_path)
    threshold = np.percentile(data_simar.loc[:, 'Hm0'], threshold_percentile)

    # Storm cycles calculation
    cycles, calm_periods = extremal.extreme_events(data_simar, 'Hm0', threshold, minimum_interarrival_time,
                                                   minimum_cycle_length, interpolation, interpolation_method,
                                                   interpolation_freq, truncate, extra_info)
    # Peaks over threshold
    peaks_over_thres = extremal.events_max(cycles)

    # Calculation of the annual maxima sample
    annual_maxima = extremal.annual_maxima_calculation(data_simar['Hm0'])

    # POT Empirical distribution
    ecdf_pot = empirical_distributions.ecdf_histogram(peaks_over_thres)
    n_peaks_year = len(peaks_over_thres) / len(data_simar['Hm0'].index.year.unique())
    ecdf_pot_rp = extremal.return_period_curve(n_peaks_year, ecdf_pot)

    # Annual Maxima Empirical distribution
    ecdf_am = empirical_distributions.ecdf_histogram(annual_maxima)
    ecdf_am_rp = extremal.return_period_curve(1, ecdf_am)

    # Fit Annual Maxima to GEV
    (param, x_gev, y_gev, y_gev_rp) = extremal.extremal_distribution_fit(data=data_simar, var_name='Hm0',
                                                                         sample=annual_maxima,
                                                                         threshold=None, fit_type='gev',
                                                                         x_min=0.90*min(annual_maxima),
                                                                         x_max=1.5*max(annual_maxima),
                                                                         n_points=1000,
                                                                         cumulative=True)

    # Fit Peaks over threshold to Poisson Pareto
    (param, x_pp, y_pp, y_pp_rp) = extremal.extremal_distribution_fit(data=data_simar, var_name='Hm0',
                                                                      sample=peaks_over_thres,
                                                                      threshold=threshold, fit_type='poisson',
                                                                      x_min=0.90*min(peaks_over_thres),
                                                                      x_max=1.5*max(peaks_over_thres),
                                                                      n_points=1000,
                                                                      cumulative=True)

    # Represent results
    plt.figure()
    ax = plt.axes()
    ax.plot(ecdf_am.index, ecdf_am, '.k', label='Annual maxima ECDF')
    ax.plot(x_gev, y_gev, 'k', label='GEV fit')
    ax.plot(x_pp, y_pp, label='Poisson-Pareto fit')
    plt.xlabel('Hm0 (m)')
    plt.ylabel('CDF')
    ax.legend()
    plt.grid()
    plt.show()

    plt.figure()
    ax = plt.axes()
    ax.semilogx(ecdf_am_rp, ecdf_am_rp.index, '.k', label='Annual maxima ECDF')
    ax.semilogx(y_gev_rp, x_gev, 'k', label='GEV fit')
    ax.semilogx(y_pp_rp, x_pp, label='Poisson-Pareto fit')
    plt.xlim(0, 500)
    plt.xlabel('Return Period (years)')
    plt.ylabel('Hm0 (m)')
    ax.legend()
    plt.grid()
    plt.show()


def test_gpd_fit_to_pot_confidence_bands():
    # Inputs
    data_file = 'SIMAR_1052046'
    threshold_percentile = 95
    minimum_interarrival_time = pd.Timedelta('3 days')
    minimum_cycle_length = pd.Timedelta('3 hours')
    interpolation = True
    interpolation_method = 'linear'
    interpolation_freq = '1min'
    truncate = False
    extra_info = False
    n_sim_boot = 100
    alpha = 0.05  # Confidence level

    # Read SIMAR
    full_data_path = os.path.join('..', '..', '..', '..', '..', 'data', 'simar')
    data_simar, code = read.simar(data_file, path=full_data_path)
    threshold = np.percentile(data_simar.loc[:, 'Hm0'], threshold_percentile)

    # Storm cycles calculation
    cycles, calm_periods = extremal.extreme_events(data_simar, 'Hm0', threshold, minimum_interarrival_time,
                                                   minimum_cycle_length, interpolation, interpolation_method,
                                                   interpolation_freq, truncate, extra_info)
    # Peaks over threshold
    peaks_over_thres = extremal.events_max(cycles)

    # POT Empirical distribution
    ecdf_pot = empirical_distributions.ecdf_histogram(peaks_over_thres)
    n_peaks_year = len(peaks_over_thres) / len(data_simar['Hm0'].index.year.unique())
    ecdf_pot_rp = extremal.return_period_curve(n_peaks_year, ecdf_pot)

    # Fit POT to Scipy-GPD
    (param_orig, x_gpd, y_gpd, y_gpd_rp) = extremal.extremal_distribution_fit(data=data_simar, var_name='Hm0',
                                                                              sample=peaks_over_thres,
                                                                              threshold=threshold, fit_type='gpd',
                                                                              x_min=0.90*min(peaks_over_thres),
                                                                              x_max=1.5*max(peaks_over_thres),
                                                                              n_points=1000,
                                                                              cumulative=True)
    # Add confidence bands to asses the uncertainty (Bootstrapping)
    boot_extreme = extremal.extremal_distribution_fit_bootstrapping(sample=peaks_over_thres,
                                                                    n_sim_boot=n_sim_boot,
                                                                    data=data_simar,
                                                                    var_name='Hm0',
                                                                    threshold=threshold,
                                                                    param_orig=param_orig,
                                                                    fit_type='gpd',
                                                                    x_min=0.90 * min(peaks_over_thres),
                                                                    x_max=1.5 * max(peaks_over_thres),
                                                                    alpha=alpha)

    # Representation
    extremal.plot_extremal_cdf(x_gpd, y_gpd, ecdf_pot, n_sim_boot, boot_extreme, alpha, title='', var_name='Hm0',
                               var_unit='m', fig_filename='', circular=False,
                               extremal_label='GPD Fit', empirical_label='POT ECDF')

    extremal.plot_extremal_return_period(x_gpd, y_gpd_rp, ecdf_pot_rp, n_sim_boot, boot_extreme, alpha,
                                         title='', var_name='Hm0', var_unit='m', fig_filename='', circular=False,
                                         extremal_label='GPD Fit', empirical_label='POT ECDF')


def test_gpd_fit_to_pot_confidence_bands_israel_era5():
    # Inputs
    threshold_percentile = 95
    minimum_interarrival_time = pd.Timedelta('3 days')
    minimum_cycle_length = pd.Timedelta('3 hours')
    interpolation = True
    interpolation_method = 'linear'
    interpolation_freq = '1min'
    truncate = False
    extra_info = False
    n_sim_boot = 10
    alpha = 0.05  # Confidence level

    # Read MODF
    location = 'israel_north'
    drivers = ['wave']

    data = []
    # Data
    for driver in drivers:
        modf = os.path.join(tests.current_path, '..', '..', 'inputadapter',  'tests', 'output', 'modf',
                            '{}_{}.modf'.format(location, driver))
        data.append(MetOceanDF.read_file(modf))

    data_simar = pd.DataFrame(data[0])

    threshold = np.percentile(data_simar.loc[:, 'swh'], threshold_percentile)

    # Storm cycles calculation
    cycles, calm_periods = extremal.extreme_events(data_simar, 'swh', threshold, minimum_interarrival_time,
                                                   minimum_cycle_length, interpolation, interpolation_method,
                                                   interpolation_freq, truncate, extra_info)
    # Peaks over threshold
    peaks_over_thres = extremal.events_max(cycles)

    # POT Empirical distribution
    ecdf_pot = empirical_distributions.ecdf_histogram(peaks_over_thres)
    n_peaks_year = len(peaks_over_thres) / len(data_simar['swh'].index.year.unique())
    ecdf_pot_rp = extremal.return_period_curve(n_peaks_year, ecdf_pot)

    # Fit POT to Scipy-GPD
    (param_orig, x_gpd, y_gpd, y_gpd_rp) = extremal.extremal_distribution_fit(data=data_simar, var_name='swh',
                                                                              sample=peaks_over_thres,
                                                                              threshold=threshold, fit_type='gpd',
                                                                              x_min=0.90*min(peaks_over_thres),
                                                                              x_max=1.5*max(peaks_over_thres),
                                                                              n_points=1000,
                                                                              cumulative=True)
    # Add confidence bands to asses the uncertainty (Bootstrapping)
    boot_extreme = extremal.extremal_distribution_fit_bootstrapping(sample=peaks_over_thres,
                                                                    n_sim_boot=n_sim_boot,
                                                                    data=data_simar,
                                                                    var_name='swh',
                                                                    threshold=threshold,
                                                                    param_orig=param_orig,
                                                                    fit_type='gpd',
                                                                    x_min=0.90 * min(peaks_over_thres),
                                                                    x_max=1.5 * max(peaks_over_thres),
                                                                    alpha=alpha)

    # Representation
    extremal.plot_extremal_cdf(x_gpd, y_gpd, ecdf_pot, n_sim_boot, boot_extreme, alpha, title='', var_name='swh',
                               var_unit='m', fig_filename='', circular=False,
                               extremal_label='GPD Fit', empirical_label='POT ECDF')

    extremal.plot_extremal_return_period(x_gpd, y_gpd_rp, ecdf_pot_rp, n_sim_boot, boot_extreme, alpha,
                                         title='', var_name='swh', var_unit='m', fig_filename='', circular=False,
                                         extremal_label='GPD Fit', empirical_label='POT ECDF')


def test_gev_fit_to_annual_maxima_confidence_bands():
    # Inputs
    data_file = 'SIMAR_1052046'
    threshold_percentile = 95
    n_sim_boot = 100
    alpha = 0.05  # Confidence level

    # Read SIMAR
    full_data_path = os.path.join('..', '..', '..', '..', '..', 'data', 'simar')
    data_simar, code = read.simar(data_file, path=full_data_path)
    threshold = np.percentile(data_simar.loc[:, 'Hm0'], threshold_percentile)

    # Calculation of the annual maxima sample
    annual_maxima = extremal.annual_maxima_calculation(data_simar['Hm0'])

    # Annual Maxima Empirical distribution
    ecdf_am = empirical_distributions.ecdf_histogram(annual_maxima)
    ecdf_am_rp = extremal.return_period_curve(1, ecdf_am)

    # Fit Annual Maxima to GEV
    (param_orig, x_gev, y_gev, y_gev_rp) = extremal.extremal_distribution_fit(data=data_simar, var_name='Hm0',
                                                                              sample=annual_maxima,
                                                                              threshold=threshold, fit_type='gev',
                                                                              x_min=0.90*min(annual_maxima),
                                                                              x_max=1.5*max(annual_maxima),
                                                                              n_points=1000,
                                                                              cumulative=True)
    # Add confidence bands to asses the uncertainty (Bootstrapping)
    boot_extreme = extremal.extremal_distribution_fit_bootstrapping(sample=annual_maxima,
                                                                    n_sim_boot=n_sim_boot,
                                                                    data=data_simar,
                                                                    var_name='Hm0',
                                                                    threshold=threshold,
                                                                    param_orig=param_orig,
                                                                    fit_type='gev',
                                                                    x_min=0.90 * min(
                                                                    annual_maxima),
                                                                    x_max=1.5 * max(annual_maxima),
                                                                    alpha=alpha)

    # Representation
    extremal.plot_extremal_cdf(x_gev, y_gev, ecdf_am, n_sim_boot, boot_extreme, alpha, title='', var_name='Hm0',
                               var_unit='m', fig_filename='', circular=False,
                               extremal_label='GEV Fit', empirical_label='GEV ECDF')

    extremal.plot_extremal_return_period(x_gev, y_gev_rp, ecdf_am_rp, n_sim_boot, boot_extreme, alpha,
                                         title='', var_name='Hm0', var_unit='m', fig_filename='', circular=False,
                                         extremal_label='GEV Fit', empirical_label='GEV ECDF')

    # # Representation
    # plt.figure()
    # ax = plt.axes()
    #
    # for sim in tqdm(range(n_sim_boot)):
    #     ax.plot(x_gev, boot_extreme['y_boot'][sim], 'grey', alpha=0.35)
    #     ax.plot(boot_extreme['ecdf_boot'][sim].index, boot_extreme['ecdf_boot'][sim], '.k', alpha=0.35)
    #
    # ax.plot(ecdf_am.index, ecdf_am, 'r.', label='Annual Maxima ECDF')
    # ax.plot(x_gev, y_gev, label='GEV Fit')
    # ax.plot(x_gev, boot_extreme['lower_band'], '--r', label='Lower 95% confidence band')
    # ax.plot(x_gev, boot_extreme['upper_band'], '--r', label='Upper 95% confidence band')
    # plt.xlabel('Hm0 (m)')
    # plt.ylabel('CDF')
    # ax.legend()
    # plt.show()
    #
    # # Representation
    # plt.figure()
    # ax = plt.axes()
    #
    # for sim in tqdm(range(n_sim_boot)):
    #     ax.semilogx(boot_extreme['y_boot_rp'][sim], x_gev, 'grey', alpha=0.35)
    #     ax.semilogx(boot_extreme['ecdf_boot_rp'][sim], boot_extreme['ecdf_boot_rp'][sim].index, '.k', alpha=0.35)
    #
    # ax.semilogx(ecdf_am_rp, ecdf_am_rp.index, '.r', label='Annual Maxima ECDF')
    # ax.semilogx(y_gev_rp, x_gev, label='GEV Fit')
    # ax.semilogx(boot_extreme['upper_band_rp'], x_gev, '--r', label='Lower 95% confidence band')
    # ax.semilogx(boot_extreme['lower_band_rp'], x_gev, '--r', label='Upper 95% confidence band')
    # plt.xlim(0, 500)
    # ax.legend()
    # plt.xlabel('Return Period (years)')
    # plt.ylabel('Hm0 (m)')
    # plt.show()
