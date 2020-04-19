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

from climate import summary
from report.sections.common import input_tex
from report.util.template import get_output_name, save_table, get_key
from report.util.driver import extract_data


def output_table_summary(modf, info, output_path):
    elements = []

    # Section title
    default_title = _('Summary table')

    # Required values
    location = info['location_metocean']
    driver = info['name_driver']
    descriptor = info['name_descriptor']
    block = info['name_block']
    section = info['name_section']

    # Optional values and default values
    title = get_key(info, 'title_section', default_title)

    # Input tex section
    input_tex(elements, info, output_path, section)

    # Computation
    data = extract_data(modf, descriptor_name=descriptor)

    # Table
    kind = 'table'
    default_caption = _('Summary table') + ': {}'.format(info['title_descriptor'])
    caption = get_key(info, 'summary_table_caption_section', default_caption)

    path = get_output_name(location=location, driver=driver, name=descriptor, block=block, title=section, kind=kind)

    data_summary = summary.get_summary(data)
    save_table(data_summary, os.path.join(output_path, path))

    elements.append([os.path.join(output_path, path), kind, caption])

    return pd.DataFrame(elements, columns=['path', 'kind', 'caption']), title


def output_plot_series(modf, info, output_path):
    elements = []

    # Section title
    default_title = _('Plot time series')

    # Required values
    location = info['location_metocean']
    driver = info['name_driver']
    descriptor = info['name_descriptor']
    block = info['name_block']
    section = info['name_section']

    # Optional values and default values
    title = get_key(info, 'title_section', default_title)
    var_name = get_key(info, 'var_name_descriptor', descriptor)
    var_unit = get_key(info, 'unit_descriptor', '')
    circular = get_key(info, 'circular_descriptor', False)

    # Input tex section
    input_tex(elements, info, output_path, section)

    # Computation
    data = extract_data(modf, descriptor_name=descriptor)

    # Figure
    kind = 'figure'
    default_caption = _('Time series plot') + ': {}'.format(info['title_descriptor'])
    caption = get_key(info, 'time_series_figure_caption_section', default_caption)

    path = get_output_name(location=location, driver=driver, name=descriptor, block=block, title=section, kind=kind)

    summary.plot_series(data, title='', var_name=var_name, var_unit=var_unit,
                        fig_filename=os.path.join(output_path, path), circular=circular)

    elements.append([path, kind, caption])

    return pd.DataFrame(elements, columns=['path', 'kind', 'caption']), title


def output_plot_series_period_time(modf, info, output_path):
    elements = []

    # Section title
    default_title = _('Plot time series')

    # Required values
    location = info['location_metocean']
    driver = info['name_driver']
    descriptor = info['name_descriptor']
    block = info['name_block']
    section = info['name_section']
    initial_date = info['initial_date_section']
    final_date = info['final_date_section']

    # Optional values and default values
    title = get_key(info, 'title_section', default_title)
    var_name = get_key(info, 'var_name_descriptor', descriptor)
    var_unit = get_key(info, 'unit_descriptor', '')
    circular = get_key(info, 'circular_descriptor', False)

    # Input tex section
    input_tex(elements, info, output_path, section)

    # Computation
    data = extract_data(modf, descriptor_name=descriptor)
    mask = ((data.index > initial_date) & (data.index < final_date))
    data_crop = data[mask]

    # Figure
    kind = 'figure'
    default_caption = _('Time series plot') + ': {}'.format(info['title_descriptor'])
    caption = get_key(info, 'time_series_figure_caption_section', default_caption)

    path = get_output_name(location=location, driver=driver, name=descriptor, block=block, title=section, kind=kind)

    summary.plot_series(data_crop, title='', var_name=var_name, var_unit=var_unit,
                        fig_filename=os.path.join(output_path, path), circular=circular)

    elements.append([path, kind, caption])

    return pd.DataFrame(elements, columns=['path', 'kind', 'caption']), title


def output_plot_histogram(modf, info, output_path):
    elements = []

    # Section title
    default_title = _('Plot histogram')

    # Required values
    location = info['location_metocean']
    driver = info['name_driver']
    descriptor = info['name_descriptor']
    block = info['name_block']
    section = info['name_section']

    # Optional values and default values
    title = get_key(info, 'title_section', default_title)
    var_name = get_key(info, 'var_name_descriptor', descriptor)
    var_unit = get_key(info, 'unit_descriptor', '')
    circular = get_key(info, 'circular_descriptor', False)
    kernel = get_key(info, 'kernel_section', False)
    bins = get_key(info, 'bins_section', 'auto')

    # Input tex section
    input_tex(elements, info, output_path, section)

    # Computation
    data = extract_data(modf, descriptor_name=descriptor)

    # Figure
    kind = 'figure'
    default_caption = _('Histogram plot') + ': {}'.format(info['title_descriptor'])
    caption = get_key(info, 'histogram_figure_caption_section', default_caption)

    path = get_output_name(location=location, driver=driver, name=descriptor, block=block, title=section, kind=kind)

    summary.plot_histogram(data, title='', var_name=var_name, var_unit=var_unit,
                           fig_filename=os.path.join(output_path, path), circular=circular, kernel=kernel, bins=bins)

    elements.append([path, kind, caption])

    return pd.DataFrame(elements, columns=['path', 'kind', 'caption']), title


def output_plot_annual_variability(modf, info, output_path):
    elements = []

    # Section title
    default_title = _('Boxplot annual variability')

    # Required values
    location = info['location_metocean']
    driver = info['name_driver']
    descriptor = info['name_descriptor']
    block = info['name_block']
    section = info['name_section']
    variability = 'year'

    # Optional values and default values
    title = get_key(info, 'title_section', default_title)
    var_name = get_key(info, 'var_name_descriptor', descriptor)
    var_unit = get_key(info, 'unit_descriptor', '')
    circular = get_key(info, 'circular_descriptor', False)
    x_step = get_key(info, 'x_step_section', 1)

    # Input tex section
    input_tex(elements, info, output_path, section)

    # Computation
    data = extract_data(modf, descriptor_name=descriptor)

    # Figure
    kind = 'figure'
    default_caption = _('Boxplot annual variability') + ': {}'.format(info['title_descriptor'])
    caption = get_key(info, 'variability_figure_caption_section', default_caption)

    path = get_output_name(location=location, driver=driver, name=descriptor, block=block, title=section, kind=kind)

    summary.plot_variability(data, frecuency=variability, title='', var_name=var_name, var_unit=var_unit,
                             fig_filename=os.path.join(output_path, path), circular=circular, x_step=x_step)

    elements.append([path, kind, caption])

    return pd.DataFrame(elements, columns=['path', 'kind', 'caption']), title


def output_plot_monthly_variability(modf, info, output_path):
    elements = []

    # Section title
    default_title = _('Boxplot monthly variability')

    # Required values
    location = info['location_metocean']
    driver = info['name_driver']
    descriptor = info['name_descriptor']
    block = info['name_block']
    section = info['name_section']
    variability = 'month'

    # Optional values and default values
    title = get_key(info, 'title_section', default_title)
    var_name = get_key(info, 'var_name_descriptor', descriptor)
    var_unit = get_key(info, 'unit_descriptor', '')
    circular = get_key(info, 'circular_descriptor', False)
    x_step = get_key(info, 'x_step_section', 1)

    # Input tex section
    input_tex(elements, info, output_path, section)

    # Computation
    data = extract_data(modf, descriptor_name=descriptor)

    # Figure
    kind = 'figure'
    default_caption = _('Boxplot monthly variability') + ': {}'.format(info['title_descriptor'])
    caption = get_key(info, 'variability_figure_caption_section', default_caption)

    path = get_output_name(location=location, driver=driver, name=descriptor, block=block, title=section, kind=kind)

    summary.plot_variability(data, frecuency=variability, title='', var_name=var_name, var_unit=var_unit,
                             fig_filename=os.path.join(output_path, path), circular=circular, x_step=x_step)

    elements.append([path, kind, caption])

    return pd.DataFrame(elements, columns=['path', 'kind', 'caption']), title


def output_plot_daily_variability(modf, info, output_path):
    elements = []

    # Section title
    default_title = _('Boxplot daily variability')

    # Required values
    location = info['location_metocean']
    driver = info['name_driver']
    descriptor = info['name_descriptor']
    block = info['name_block']
    section = info['name_section']
    variability = 'dayofyear'

    # Optional values and default values
    title = get_key(info, 'title_section', default_title)
    var_name = get_key(info, 'var_name_descriptor', descriptor)
    var_unit = get_key(info, 'unit_descriptor', '')
    circular = get_key(info, 'circular_descriptor', False)
    x_step = get_key(info, 'x_step_section', 1)

    # Input tex section
    input_tex(elements, info, output_path, section)

    # Computation
    data = extract_data(modf, descriptor_name=descriptor)

    # Figure
    kind = 'figure'
    default_caption = _('Boxplot daily variability') + ': {}'.format(info['title_descriptor'])
    caption = get_key(info, 'variability_figure_caption_section', default_caption)

    path = get_output_name(location=location, driver=driver, name=descriptor, block=block, title=section, kind=kind)

    summary.plot_variability(data, frecuency=variability, title='', var_name=var_name, var_unit=var_unit,
                             fig_filename=os.path.join(output_path, path), circular=circular, x_step=x_step)

    elements.append([path, kind, caption])

    return pd.DataFrame(elements, columns=['path', 'kind', 'caption']), title


def output_plot_variability_period_time(modf, info, output_path):
    elements = []

    # Section title
    default_title = _('Plot variability')

    # Required values
    location = info['location_metocean']
    driver = info['name_driver']
    descriptor = info['name_descriptor']
    block = info['name_block']
    section = info['name_section']
    variability = info['variability_section']
    initial_date = info['initial_date_section']
    final_date = info['final_date_section']

    # Optional values and default values
    title = get_key(info, 'title_section', default_title)
    var_name = get_key(info, 'var_name_descriptor', descriptor)
    var_unit = get_key(info, 'unit_descriptor', '')
    circular = get_key(info, 'circular_descriptor', False)
    x_step = get_key(info, 'x_step_section', 1)

    # Input tex section
    input_tex(elements, info, output_path, section)

    # Computation
    data = extract_data(modf, descriptor_name=descriptor)
    mask = ((data.index > initial_date) & (data.index < final_date))
    data_crop = data[mask]

    # Figure
    kind = 'figure'
    default_caption = _('Variability boxplot') + ': {}'.format(info['title_descriptor'])
    caption = get_key(info, 'variability_figure_caption_section', default_caption)

    path = get_output_name(location=location, driver=driver, name=descriptor, block=block, title=section, kind=kind)

    summary.plot_variability(data_crop, frecuency=variability, title='', var_name=var_name, var_unit=var_unit,
                             fig_filename=os.path.join(output_path, path), circular=circular, x_step=x_step)

    elements.append([path, kind, caption])

    return pd.DataFrame(elements, columns=['path', 'kind', 'caption']), title
