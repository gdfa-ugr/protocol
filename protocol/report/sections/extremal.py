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
import pandas as pd
import numpy as np

from climate.stats import extremal, empirical_distributions
from report.sections.common import input_tex
from report.util.template import get_output_name, save_table, get_key
from report.util.driver import extract_data


def output_plot_gpd_fit_peaks_over_threshold(modf, info, output_path):
    elements = []

    # Section title
    default_title = _('Plot gpd fit to peaks over threshold')

    # Required values
    location = info['location_metocean']
    driver = info['name_driver']
    descriptor = info['name_descriptor']
    block = info['name_block']
    section = info['name_section']
    threshold_value = info['percentile_threshold_section']
    minimum_interarrival_time = info['minimum_interarrival_time_section']
    minimum_cycle_length = info['minimum_cycle_length_section']
    n_sim_boot = info['n_sim_bootstrapping_section']
    alpha = info['alpha_section']

    # Optional values and default values
    title = get_key(info, 'title_section', default_title)
    var_name = get_key(info, 'var_name_descriptor', descriptor)
    var_unit = get_key(info, 'unit_descriptor', '')
    circular = get_key(info, 'circular_descriptor', False)
    interpolation = get_key(info, 'interpolation_section', True)
    interpolation_method = get_key(info, 'interpolation_method_section', 'linear')
    interpolation_freq = get_key(info, 'interpolation_freq_section', '1min')
    truncate = get_key(info, 'truncate_section', False)
    label_empirical = get_key(info, 'label_empirical_section', '')
    label_gpd_fit = get_key(info, 'label_gpd_fit_section', '')

    # Input tex section
    input_tex(elements, info, output_path, section)

    # Computation
    minimum_interarrival_time = pd.Timedelta(minimum_interarrival_time)
    minimum_cycle_length = pd.Timedelta(minimum_cycle_length)
    # Remove null values
    modf_clean = modf.dropna(axis=0, how='all')
    threshold = np.percentile(modf_clean[descriptor], threshold_value)

    # Storm cycles calculation
    cycles, calm_periods = extremal.extreme_events(modf_clean, descriptor, threshold, minimum_interarrival_time,
                                                   minimum_cycle_length, interpolation, interpolation_method,
                                                   interpolation_freq, truncate, extra_info=False)
    # Peaks over threshold
    peaks_over_thres = extremal.events_max(cycles)

    # POT Empirical distribution
    ecdf_pot = empirical_distributions.ecdf_histogram(peaks_over_thres)
    n_peaks_year = len(peaks_over_thres) / len(modf.index.year.unique())
    ecdf_pot_rp = extremal.return_period_curve(n_peaks_year, ecdf_pot)

    # Fit POT to Scipy-GPD
    (param_orig, x_gpd, y_gpd, y_gpd_rp) = extremal.extremal_distribution_fit(data=modf_clean, var_name=descriptor,
                                                                              sample=peaks_over_thres,
                                                                              threshold=threshold, fit_type='gpd',
                                                                              x_min=0.90*min(peaks_over_thres),
                                                                              x_max=3*max(peaks_over_thres),
                                                                              n_points=1000,
                                                                              cumulative=True)
    # Add confidence bands to asses the uncertainty (Bootstrapping)
    boot_extreme = extremal.extremal_distribution_fit_bootstrapping(sample=peaks_over_thres,
                                                                    n_sim_boot=n_sim_boot,
                                                                    data=modf_clean,
                                                                    var_name=descriptor,
                                                                    threshold=threshold,
                                                                    param_orig=param_orig,
                                                                    fit_type='gpd',
                                                                    x_min=0.90 * min(peaks_over_thres),
                                                                    x_max=3 * max(peaks_over_thres),
                                                                    alpha=alpha)

    # Figure
    kind = 'figure'
    default_caption = _('GPD fit to peaks over threshold plot') + ': {}'.format(info['title_descriptor'])
    caption = get_key(info, 'gpd_fit_figure_caption_section', default_caption)

    path = get_output_name(location=location, driver=driver, name=descriptor, block=block, title=section, kind=kind)

    extremal.plot_extremal_cdf(x_gpd, y_gpd, ecdf_pot, n_sim_boot, boot_extreme, alpha, title='', var_name=var_name,
                               var_unit=var_unit, fig_filename=os.path.join(output_path, path), circular=circular,
                               extremal_label=label_gpd_fit, empirical_label=label_empirical)

    elements.append([path, kind, caption])

    # Figure
    kind = 'figure'
    default_caption = _('GPD fit to peaks over threshold plot (return period format)') + ': {}'.format(
        info['title_descriptor'])
    caption = get_key(info, 'gpd_fit_return_period_figure_caption_section', default_caption)

    path = get_output_name(location=location, driver=driver, name=descriptor, block=block, title=section+'_rp',
                           kind=kind)

    extremal.plot_extremal_return_period(x_gpd, y_gpd_rp, ecdf_pot_rp, n_sim_boot, boot_extreme, alpha,
                                         title='', var_name=var_name, var_unit=var_unit,
                                         fig_filename=os.path.join(output_path, path), circular=circular,
                                         extremal_label=label_gpd_fit, empirical_label=label_empirical)

    elements.append([path, kind, caption])

    return pd.DataFrame(elements, columns=['path', 'kind', 'caption']), title


def output_plot_gev_fit_annual_maxima(modf, info, output_path):
    elements = []

    # Section title
    default_title = _('Plot gev fit to annual maxima')

    # Required values
    location = info['location_metocean']
    driver = info['name_driver']
    descriptor = info['name_descriptor']
    block = info['name_block']
    section = info['name_section']
    n_sim_boot = info['n_sim_bootstrapping_section']
    alpha = info['alpha_section']

    # Optional values and default values
    title = get_key(info, 'title_section', default_title)
    var_name = get_key(info, 'var_name_descriptor', descriptor)
    var_unit = get_key(info, 'unit_descriptor', '')
    circular = get_key(info, 'circular_descriptor', False)
    label_empirical = get_key(info, 'label_empirical_section', '')
    label_gev_fit = get_key(info, 'label_gev_fit_section', '')

    # Input tex section
    input_tex(elements, info, output_path, section)

    # Computation
    # Calculation of the annual maxima sample
    annual_maxima = extremal.annual_maxima_calculation(modf[descriptor])

    # Annual Maxima Empirical distribution
    ecdf_am = empirical_distributions.ecdf_histogram(annual_maxima)
    ecdf_am_rp = extremal.return_period_curve(1, ecdf_am)

    # Fit Annual Maxima to GEV
    (param_orig, x_gev, y_gev, y_gev_rp) = extremal.extremal_distribution_fit(data=modf, var_name=descriptor,
                                                                              sample=annual_maxima,
                                                                              threshold=None, fit_type='gev',
                                                                              x_min=0.90*min(annual_maxima),
                                                                              x_max=3*max(annual_maxima),
                                                                              n_points=1000,
                                                                              cumulative=True)
    # Add confidence bands to asses the uncertainty (Bootstrapping)
    boot_extreme = extremal.extremal_distribution_fit_bootstrapping(sample=annual_maxima,
                                                                    n_sim_boot=n_sim_boot,
                                                                    data=modf,
                                                                    var_name=descriptor,
                                                                    threshold=None,
                                                                    param_orig=param_orig,
                                                                    fit_type='gev',
                                                                    x_min=0.90 * min(annual_maxima),
                                                                    x_max=3 * max(annual_maxima),
                                                                    alpha=alpha)

    # Figure
    kind = 'figure'
    default_caption = _('GEV fit to annual maxima plot') + ': {}'.format(info['title_descriptor'])
    caption = get_key(info, 'gev_fit_figure_caption_section', default_caption)

    path = get_output_name(location=location, driver=driver, name=descriptor, block=block, title=section, kind=kind)

    # Representation
    extremal.plot_extremal_cdf(x_gev, y_gev, ecdf_am, n_sim_boot, boot_extreme, alpha, title='', var_name=var_name,
                               var_unit=var_unit, fig_filename=os.path.join(output_path, path), circular=circular,
                               extremal_label=label_gev_fit, empirical_label=label_empirical)

    elements.append([path, kind, caption])

    # Figure
    kind = 'figure'
    default_caption = _('GEV fit to annual maxima plot (return period format)') + ': {}'.format(
        info['title_descriptor'])
    caption = get_key(info, 'gev_fit_return_period_figure_caption_section', default_caption)

    path = get_output_name(location=location, driver=driver, name=descriptor, block=block, title=section+'_rp',
                           kind=kind)

    # Representation
    extremal.plot_extremal_return_period(x_gev, y_gev_rp, ecdf_am_rp, n_sim_boot, boot_extreme, alpha,
                                         title='', var_name=var_name, var_unit=var_unit,
                                         fig_filename=os.path.join(output_path, path), circular=circular,
                                         extremal_label=label_gev_fit, empirical_label=label_empirical)

    elements.append([path, kind, caption])

    return pd.DataFrame(elements, columns=['path', 'kind', 'caption']), title
