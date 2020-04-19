#!/usr/bin/env python
# coding: utf-8
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

try:
    # noinspection PyUnresolvedReferences, PyCompatibility
    from builtins import *  # noqa
except ImportError:
    pass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from scipy.stats import genpareto
from scipy.stats import genextreme
from scipy.stats import exponweib
from climate.util import plot
from climate.util import time_series
from climate.stats.empirical_distributions import ecdf_histogram


def annual_maxima_calculation(data):
    # TODO Mejorar implmentacion para devolver la fecha exacta de maximo
    # Aggregation of the hourly data by day
    data_aggregated = data.resample('D').max()
    # Stack the data by year
    simar_year_stack = time_series.generate_year_stack(data_aggregated)
    # Calculation of the maximum of each year
    annual_maxima = simar_year_stack.max(axis=1)
    annual_maxima.index = data.index.year.unique()

    return annual_maxima


def peaks_plot(data, peaks, data_column='',  title='', var_name='', var_unit='', fig_filename='',
               circular=False, data_label='', peaks_label=''):
    # Get style
    plot.get_default_plot_style()
    # Get values
    values = plot.get_values(data, data_column)
    # Get var label
    var_label = plot.get_var_label(var_name, var_unit)
    # Plot
    ax = plt.axes()
    ax.plot(values.index, values, ':', alpha=0.3, label=data_label)
    # ax.plot(peaks.index, peaks, '.r', markersize=10, label=peaks_label)
    ax.legend(loc='upper right', facecolor='white')
    plot.plot_title(ax, title)


    if circular:
        circular_labels, circular_ticks = plot.get_circular_ticks()
        plt.yticks(circular_ticks, circular_labels)

    ax.set_ylabel(var_label)
    # fixing the x tick labels are all squashed together
    plt.gcf().autofmt_xdate()
    plt.xlim([1959, 2018])

    plot.save_figure(fig_filename)


def missing_values_plot(data, data_gaps, data_column='', title='', var_name='', var_unit='', fig_filename='',
                        circular=False, label='observations'):
    # Get style
    plot.get_default_plot_style()
    # Get values
    values = plot.get_values(data, data_column)
    # Get var label
    var_label = plot.get_var_label(var_name, var_unit)
    # Plot
    ax = plt.axes()
    ax.plot(values.index, values, ':', alpha=0.3, label=label)
    ax.plot(values[data_gaps.loc[:, 'pos_ini']], '.k', markersize=10, label='Gaps_initial_postion')
    ax.plot(values[data_gaps.loc[:, 'pos_fin']], '.r', markersize=10, label='Gaps_final_postion')
    ax.legend(loc='upper right', facecolor='white')
    plot.plot_title(ax, title)

    if circular:
        circular_labels, circular_ticks = plot.get_circular_ticks()
        plt.yticks(circular_ticks, circular_labels)

    ax.set_ylabel(var_label)
    # fixing the x tick labels are all squashed together
    plt.gcf().autofmt_xdate()

    plot.save_figure(fig_filename)


def extreme_events(data, var_name, threshold, minimum_interarrival_time, minimum_cycle_length, interpolation=False,
                   interpolation_method='linear', interpolation_freq='1min', truncate=False, extra_info=False):
    """ Extract storm and calm cycles from a time series data

    Args:
        data (pd.DataFrame): Time series data
        var_name (str): Name of the data column variable
        threshold (int): Threshold that separates storm (cycle) and calm periods
        minimum_interarrival_time (pd.Timedelta): Minimum time between storms to guarantee that these storms are
            independent
        minimum_cycle_length (pd.Timedelta): Minimum duration of a storm to considet it valid
        interpolation (bool, optional): Enabling or disabling the interpolation method to estimate the time between
            the instant when the thresolhold is reached and the instant when storm starts. When cross tiem parameter
            is set to True the cross time method is used to improved the duration of the cycle with disregard of
            the value that takes this parameter
        interpolation_method (str, optinal): Interpolation method
        interpolation_freq (str, optional):
        truncate (bool, optional): If it is enable, the values of the storms lower than the threshold and the values of
            the calms higher than the threshold are set equal to the threshold
        extra_info (bool, optional):

    Returns:
        tuple:

        - (np.array): Indices with the position of the start of the storm cycles
        - (np.array): Indices with the position of the end of the storm cycles
    """
    full_data = data.copy()
    # noinspection PyUnresolvedReferences
    var_data = full_data[var_name]
    cycles_start, cycles_end = time_series.values_over_threshold(var_data, threshold)

    # If data ends with a cycle, it is necessary to remove it
    # noinspection PyTypeChecker
    if len(var_data.index) == cycles_end[-1]:
        var_data = var_data[:cycles_start[-1]]
        cycles_start = cycles_start[:-1]
        cycles_end = cycles_end[:-1]

    new_cycles_limits_indexes = None
    if interpolation:
        new_cycles_limits_indexes = time_series.interpolation_boundaries_index(var_data, cycles_start, cycles_end,
                                                                               interpolation_freq, interpolation_method,
                                                                               threshold)
        new_cycles_limits = pd.DataFrame(threshold, index=new_cycles_limits_indexes, columns=[var_name])

        # noinspection PyUnresolvedReferences
        full_data = full_data.combine_first(new_cycles_limits)
        full_data = full_data.interpolate(interpolation_method)

        var_data = full_data[var_name]
        cycles_start, cycles_end = time_series.values_over_threshold(var_data, threshold)

    # Find near cycles (not independent cycles)
    near_cycles = time_series.near_events(var_data, cycles_start, cycles_end, minimum_interarrival_time, interpolation)

    while np.any(near_cycles):
        # Join near cycles
        cycles_start = cycles_start[np.append(True, ~near_cycles)]
        cycles_end = cycles_end[np.append(~near_cycles, True)]

        # Find if there are more near cycles
        near_cycles = time_series.near_events(var_data, cycles_start, cycles_end, minimum_interarrival_time,
                                              interpolation)

    # Remove short cycles
    cycles_length = var_data.index[cycles_end - 1] - var_data.index[cycles_start]
    short_cycles = cycles_length < minimum_cycle_length

    cycles_start = cycles_start[~short_cycles]
    cycles_end = cycles_end[~short_cycles]

    cycles_indexes_clipped = None
    calm_periods_indexes_clipped = None
    cycles_mask = time_series.extreme_indexes(var_data, cycles_start, cycles_end)
    if truncate:
        lower_values_clipped = var_data[cycles_mask].clip(lower=threshold)
        upper_values_clipped = var_data[~cycles_mask].clip(upper=threshold)

        cycles_indexes_clipped = lower_values_clipped != var_data[cycles_mask]
        calm_periods_indexes_clipped = upper_values_clipped != var_data[~cycles_mask]
        # noinspection PyUnresolvedReferences
        cycles_indexes_clipped = cycles_indexes_clipped[cycles_indexes_clipped].index.tolist()
        # noinspection PyUnresolvedReferences
        calm_periods_indexes_clipped = calm_periods_indexes_clipped[calm_periods_indexes_clipped].index.tolist()

        var_data[cycles_mask] = lower_values_clipped
        var_data[~cycles_mask] = upper_values_clipped

    # Split cycles and calm periods
    cross_indexes = np.sort(np.concatenate([cycles_start, cycles_end]))
    cross_indexes = cross_indexes[cross_indexes != 0]  # avoid splitting by 0 index

    data_splitted = np.split(var_data, cross_indexes)

    # Check if var_data starts with a cycle or a calm period
    if cycles_start[0] == 0:
        cycles = data_splitted[0::2]
        calm_periods = data_splitted[1::2]

        cycles_indexes = var_data[cycles_mask].index
        calm_periods_indexes = var_data[~cycles_mask].index
    else:
        cycles = data_splitted[1::2]
        calm_periods = data_splitted[2::2]

        cycles_indexes = var_data[cycles_mask].index
        calm_periods_indexes = var_data[~cycles_mask].index.difference(data_splitted[0].index)

    if extra_info:
        info = dict()

        info['data_cycles'] = full_data.loc[cycles_indexes]
        info['data_calm_periods'] = full_data.loc[calm_periods_indexes]

        if truncate:
            info['cycles_indexes_clipped'] = cycles_indexes_clipped
            info['calm_periods_indexes_clipped'] = calm_periods_indexes_clipped

        if interpolation:
            info['interpolation_indexes'] = new_cycles_limits_indexes

        return cycles, calm_periods, info
    else:
        return cycles, calm_periods


def events_max(events):
    """ Extract maximum value of each cycle

    Args:
        events (list):  List of storm or calm cycles where each element is a pd.Series with the cycles values and the
            time of occurrence

    Returns:
        pd.Series: Maximum value of each cycle with the associated time of occurrence
    """
    events_stacked = pd.DataFrame(events).T

    index = events_stacked.idxmax().values
    # OPTIMIZE Check if it is possible to derive values from index
    values = events_stacked.max().values

    return pd.Series(values, index=index)


def events_boundaries(events):
    """ Extract the starting and ending indexes of each cycle or calm cycles

    Args:
        events (list):  List of storm or calm cycles where each element is a pd.Series with the cycles values and the
            time of occurrence

    Returns:
        tuple:

            - (pd.Series): Starting index of each extreme event
            - (pd.Series): Ending index of each extreme event
    """
    events_stacked = pd.DataFrame(events).T

    start = events_stacked.apply(pd.Series.first_valid_index).reset_index(drop=True)
    end = events_stacked.apply(pd.Series.last_valid_index).reset_index(drop=True)

    return start, end


def events_duration(events):
    """ Extract the duration of each cycle

    Args:
        events (list):  List of storm or calm cycles where each element is a pd.Series with the cycles values and the
            time of occurrence

    Returns:
        pd.Series: Duration of each cycle with the associated time of occurrence
    """
    events_stacked = pd.DataFrame(events).T

    start = events_stacked.apply(pd.Series.first_valid_index)
    end = events_stacked.apply(pd.Series.last_valid_index)

    events_length = end - start

    duration = pd.Series(events_length.values, index=start.values)

    return duration


def events_magnitude(events, threshold, x_unit=pd.Timedelta('1 hour')):
    """ Calculate the magnitude (area under the cycle curve) of each cycle

    Args:
        events (list):  List of storm or calm cycles where each element is a pd.Series with the cycles values and the
            time of occurrence
        threshold ():
        x_unit ():

    Returns:
        np.array: Duration of each cycle with the associated time of occurrence
    """
    # OPTIMIZE Resampling and interpolation before computing the magnitude
    events_stacked = pd.DataFrame(events).T - threshold

    # Idea taken from https://berkeley-stat159-f17.github.io/stat159-f17/lectures/09-intro-numpy/trapezoid..html
    # Currently neither Numpy and SciPy trapz() does work properly with null values nor with numpy masked arrays
    dx = (events_stacked.index[1:] - events_stacked.index[:-1])[:, np.newaxis] / x_unit
    magnitude = np.nansum(dx * (events_stacked.values[1:] + events_stacked.values[:-1]), axis=0) / 2

    return magnitude


def compute_events(events, f):
    events_stacked = pd.DataFrame(events).T

    return getattr(events_stacked, f)()


def extremal_distribution_fit(data, var_name, sample, threshold, fit_type, x_min, x_max, n_points, loc=None, scale=None,
                              cumulative=True):
    # Initialization of the output variables
    param = None
    x = None
    y = None
    y_rp = None

    if fit_type == 'gpd':
        # Fit the exceedances over threshold to Generalized Pareto distribution
        param = generalized_pareto_distribution_fit(sample, threshold, loc, scale)

        # Calculate the pdf and/or cdf
        x = np.linspace(x_min, x_max, n_points)

        if cumulative:
            y = genpareto.cdf(x, param[0], param[1], param[2])

            # Calculate the number of extreme peaks per year
            n_peaks_year = len(sample) / len(data[var_name].index.year.unique())
            y_rp = return_period_curve(n_peaks_year, y)
        else:
            y = genpareto.pdf(x, param[0], param[1], param[2])

    elif fit_type == 'coles':
        # Fit the exceedances over threshold to Generalized Pareto distribution
        param = generalized_pareto_distribution_fit(sample, threshold, loc, scale)

        x = np.arange(1, 501)
        u = param[1]
        sigma = param[2]
        xi = param[0]

        # Mean number of data in a year (numero medio de datos en un año)
        n_y = len(data[var_name]) / len(data[var_name].index.year.unique())
        # Total number of POT / number of years
        z_u = len(sample) / len(data[var_name])
        # n_y*z_u is the number of POT / number of years -- > numer of POT per year
        y_rp = u + (sigma / xi) * (((x * n_y * z_u) ** xi) - 1)

    elif fit_type == 'gev':
        param = generalized_extreme_value_distribution_fit(sample, loc, scale)

        # Calculate the pdf and/or cdf
        x = np.linspace(x_min, x_max, n_points)

        if cumulative:
            y = genextreme.cdf(x, param[0], param[1], param[2])

            # Calculate the number of extreme peaks per year
            n_peaks_year = 1
            y_rp = return_period_curve(n_peaks_year, y)
        else:
            y = genpareto.pdf(x, param[0], param[1], param[2])

    elif fit_type == 'poisson':
        # Calculate the pdf and/or cdf
        x = np.linspace(x_min, x_max, n_points)

        # Fit the exceedances over threshold to Generalized Pareto distribution
        gpd_param = generalized_pareto_distribution_fit(sample, threshold, loc, scale)

        # Poisson parameter (número de eventos extraños al año)
        poisspareto_param = len(sample) / len(data[var_name].index.year.unique())
        # Poisson pareto parameters
        poisspareto_param = [poisspareto_param, gpd_param[0], gpd_param[2], gpd_param[1]]
        # Equivalent gev parameters
        param = [0, 0, 0]
        param[0] = -poisspareto_param[1]
        param[1] = poisspareto_param[2] * (poisspareto_param[0] ** poisspareto_param[1])
        param[2] = poisspareto_param[3] + (
                (poisspareto_param[2] / poisspareto_param[1]) * ((poisspareto_param[0] ** poisspareto_param[1]) - 1))

        if cumulative:
            y = genextreme.cdf(x, param[0], param[2], param[1])

            # Calculate the number of extreme peaks per year
            n_peaks_year = 1
            y_rp = return_period_curve(n_peaks_year, y)
        else:
            y = genextreme.pdf(x, param[0], param[2], param[1])

    return param, x, y, y_rp


def generalized_pareto_distribution_fit(peaks_over_th, threshold, loc=None, scale=None):
    # Fit the exceedances over threshold to Generalized Pareto distribution
    # BUG Missing default values get different results than default parameters values
    if loc is None and scale is not None:
        gpd_param = genpareto.fit(peaks_over_th - threshold, scale=scale)
    elif loc is not None and scale is None:
        gpd_param = genpareto.fit(peaks_over_th - threshold, loc=loc)
    elif loc is None and scale is None:
        gpd_param = genpareto.fit(peaks_over_th - threshold)
    else:
        gpd_param = genpareto.fit(peaks_over_th - threshold, loc=loc, scale=scale)

    # Set the localization parameter equal to the threshold
    gpd_param = list(gpd_param)
    gpd_param[1] = threshold

    return gpd_param


def generalized_extreme_value_distribution_fit(annual_maxima, loc=None, scale=None):
    # Fit the exceedances over threshold to Generalized Pareto distribution
    # BUG Missing default values get different results than default parameters values
    if loc is None and scale is not None:
        gev_param = genextreme.fit(annual_maxima, scale=scale)
    elif loc is not None and scale is None:
        gev_param = genextreme.fit(annual_maxima, loc=loc)
    elif loc is None and scale is None:
        gev_param = genextreme.fit(annual_maxima)
    else:
        gev_param = genextreme.fit(annual_maxima, loc=loc, scale=scale)

    return gev_param


def return_period_curve(n_peaks_year, cdf):
    return_period = 1 / (n_peaks_year * (1 - cdf))

    return return_period


def simulation_sample(sample, n_simulation_bootsp):

    # TODO: hacer paquete opcional
    from tqdm import tqdm

    # List of elements positions
    sample_list = np.arange(0, len(sample))

    # Generation of a new pot sample
    sample_boot = []

    for _ in tqdm(range(n_simulation_bootsp)):
        new_sample_list = np.random.choice(sample_list, len(sample_list), replace=True)
        sample_boot.append(sample.iloc[new_sample_list])

    return sample_boot


def extremal_distribution_fit_bootstrapping(sample, n_sim_boot, data, var_name, threshold, param_orig,
                                            fit_type, x_min, x_max, alpha):
    # TODO: hacer paquete opcional
    from tqdm import tqdm

    # TODO: que el input sea la función de scipy y no una string. Permite hacer más ajustes.
    # This calculation is only done for gpd and gev
    if fit_type == 'gpd' or fit_type == 'gev':
        # Calculate the number of peaks per year
        if fit_type == 'gpd':
            n_peaks_year = len(sample) / len(data[var_name].index.year.unique())
        else:
            n_peaks_year = 1

        # Generation of new samples of peaks over threshold
        peaks_boot = simulation_sample(sample, n_sim_boot)

        # Initialization
        ecdf_boot = []
        ecdf_boot_rp = []
        x_boot = []
        y_boot = []
        y_boot_rp = []

        # Fit a extremal distribution for each new sample
        for sim in tqdm(range(n_sim_boot)):
            peaks_sample_boot = pd.Series(peaks_boot[sim])

            # Fit empirical distribution to peaks over threshold
            ecdf_boot.append(ecdf_histogram(peaks_sample_boot))

            # Calculate the return period curve
            ecdf_boot_rp.append(return_period_curve(n_peaks_year, ecdf_boot[sim]))


            # Fit peaks sample to theretical distribution to obtain the parameters
            (param, x_boot_sim, y_boot_sim, y_boot_sim_rp) = extremal_distribution_fit(data=data, var_name=var_name,
                                                                                       sample=peaks_sample_boot,
                                                                                       threshold=threshold,
                                                                                       fit_type=fit_type,
                                                                                       x_min=x_min,
                                                                                       x_max=x_max,
                                                                                       n_points=1000,
                                                                                       cumulative=True)
            # # Get style
            # fig1 = plt.figure()
            # plot.get_default_plot_style()
            # ax = plt.axes()
            # ax.semilogx(y_boot_sim_rp, x_boot_sim, 'grey', alpha=0.35)
            # ax.semilogx(ecdf_boot_rp[sim], ecdf_boot_rp[sim].index, '.r')
            # plt.xlim(0, 500)
            # ax.legend()
            # fig1.savefig(
            #     os.path.join('..', 'output', 'img', 'extremal', 'Ajuste_bootstrap_' + str(sim).zfill(4) + '.png'))

            if np.max(y_boot_sim_rp) > 1000:
                # Add results to the list
                x_boot.append(x_boot_sim)
                y_boot.append(y_boot_sim)
                y_boot_rp.append(y_boot_sim_rp)
            else:
                # Add results to the list
                x_boot.append(x_boot[-1])
                y_boot.append(y_boot[-1])
                y_boot_rp.append(y_boot_rp[-1])



        # Stack peaks boot sample
        y_boot_gpd_stacked = np.stack(y_boot).T

        # Extract upper and lower confidence bounds
        upper_band = np.percentile(y_boot_gpd_stacked, (alpha / 2) * 100, axis=1)
        lower_band = np.percentile(y_boot_gpd_stacked, (1 - alpha / 2) * 100, axis=1)

        upper_band_rp = 1 / (n_peaks_year * (1 - upper_band))
        lower_band_rp = 1 / (n_peaks_year * (1 - lower_band))

        # Output dictionary
        boot_extreme = {'ecdf_boot': ecdf_boot, 'ecdf_boot_rp': ecdf_boot_rp, 'x_boot': x_boot, 'y_boot': y_boot,
                        'y_boot_rp': y_boot_rp, 'upper_band': upper_band, 'lower_band': lower_band,
                        'upper_band_rp': upper_band_rp, 'lower_band_rp': lower_band_rp}

        return boot_extreme


def plot_extremal_cdf(x_extremal, y_extremal, ecdf_extemal, n_sim_boot, boot_extremal, alpha, title='',
                      var_name='', var_unit='', fig_filename='', circular=False,
                      extremal_label='', empirical_label=''):
    # TODO: hacer paquete opcional
    from tqdm import tqdm

    # Get style
    plot.get_default_plot_style()

    # Prepare label and units
    var_label = plot.get_var_label(var_name, var_unit)

    ax = plt.axes()
    # for sim in tqdm(range(n_sim_boot)):
    #     ax.plot(x_extremal, boot_extremal['y_boot'][sim], 'grey', alpha=0.35)

    ax.plot(ecdf_extemal.index, ecdf_extemal, '.', color='#ff7f0e', label=empirical_label)
    ax.plot(x_extremal, y_extremal, color='#1f77b4', label=extremal_label)
    ax.plot(x_extremal, boot_extremal['lower_band'], '--', color='#1f77b4', label='Lower {}% confidence band'.format((1-alpha)*100))
    ax.plot(x_extremal, boot_extremal['upper_band'], '--', color='#1f77b4', label='Upper {}% confidence band'.format((1-alpha)*100))
    ax.legend()

    plot.plot_title(ax, title)

    if circular:
        circular_labels, circular_ticks = plot.get_circular_ticks()
        plt.yticks(circular_ticks, circular_labels)

    ax.set_xlabel(var_label)
    ax.set_ylabel('CDF')

    # fixing the x tick labels are all squashed together
    plt.gcf().autofmt_xdate()

    plot.save_figure(fig_filename)


def plot_extremal_return_period(x_extremal_rp, y_extremal_rp, ecdf_extemal_rp, n_sim_boot, boot_extremal, alpha,
                                title='', var_name='', var_unit='', fig_filename='', circular=False,
                                extremal_label='', empirical_label=''):
    # TODO: hacer paquete opcional
    from tqdm import tqdm

    # Get style
    plot.get_default_plot_style()

    # Prepare label and units
    var_label = plot.get_var_label(var_name, var_unit)

    ax = plt.axes()

    # for sim in tqdm(range(n_sim_boot)):
    #     ax.semilogx(boot_extremal['y_boot_rp'][sim], x_extremal_rp, 'grey', alpha=0.35)

    ax.semilogx(ecdf_extemal_rp, ecdf_extemal_rp.index, '.', color='#ff7f0e', label=empirical_label)
    ax.semilogx(y_extremal_rp, x_extremal_rp, color='#1f77b4', label=extremal_label)
    ax.semilogx(boot_extremal['upper_band_rp'], x_extremal_rp, '--', color='#1f77b4',
                label='Lower {}% confidence band'.format((1-alpha)*100))
    ax.semilogx(boot_extremal['lower_band_rp'], x_extremal_rp, '--', color='#1f77b4',
                label='Upper {}% confidence band'.format((1-alpha)*100))
    plt.xlim(0, 500)
    ax.legend()

    plot.plot_title(ax, title)

    if circular:
        circular_labels, circular_ticks = plot.get_circular_ticks()
        plt.yticks(circular_ticks, circular_labels)

    ax.set_xlabel('Return period (years)')
    ax.set_ylabel(var_label)

    # fixing the x tick labels are all squashed together
    plt.gcf().autofmt_xdate()

    plot.save_figure(fig_filename)

