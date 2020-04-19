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

from climate.stats import empirical_distributions
from report.sections.common import input_tex
from report.util.template import get_output_name, save_table, get_key
from report.util.driver import extract_data


def output_plot_empirical_pdf(modf, info, output_path):
    elements = []

    # Section title
    default_title = _('Plot empirical PDF')

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
    label_empirical = get_key(info, 'label_empirical_section', '')
    bins = get_key(info, 'bins_section', 0)

    # Input tex section
    input_tex(elements, info, output_path, section)

    # Computation
    data = extract_data(modf, descriptor_name=descriptor)
    # Remove null values
    values_clean = data.dropna(axis=0, how='all')
    data_empirical = empirical_distributions.epdf_histogram(values_clean, bins)
    cumulative = False
    data_kernel = None

    # Figure
    kind = 'figure'
    default_caption = _('Empirical PDF') + ': {}'.format(info['title_descriptor'])
    caption = get_key(info, 'empirical_pdf_figure_caption_section', default_caption)

    path = get_output_name(location=location, driver=driver, name=descriptor, block=block, title=section, kind=kind)

    empirical_distributions.plot_kde(data_empirical, data_kernel, cumulative, title='', var_name=var_name,
                                     var_unit=var_unit, fig_filename=os.path.join(output_path, path),
                                     circular=circular, label_empirical=label_empirical,
                                     label_kernel='')

    elements.append([path, kind, caption])

    return pd.DataFrame(elements, columns=['path', 'kind', 'caption']), title


def output_plot_empirical_cdf(modf, info, output_path):
    elements = []

    # Section title
    default_title = _('Plot empirical CDF')

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
    label_empirical = get_key(info, 'label_empirical_section', '')
    label_kernel = get_key(info, 'label_kernel_section', '')
    bins = get_key(info, 'bins_section', 100)

    # Input tex section
    input_tex(elements, info, output_path, section)

    # Computation
    data = extract_data(modf, descriptor_name=descriptor)
    cumulative = True
    data_empirical = empirical_distributions.ecdf_histogram(data)
    data_kernel = empirical_distributions.kde_sm(data, cumulative=cumulative, gridsize=bins)

    # Figure
    kind = 'figure'
    default_caption = _('Empirical CDF') + ': {}'.format(info['title_descriptor'])
    caption = get_key(info, 'empirical_pdf_figure_caption_section', default_caption)

    path = get_output_name(location=location, driver=driver, name=descriptor, block=block, title=section, kind=kind)

    empirical_distributions.plot_kde(data_empirical, data_kernel, cumulative, title='', var_name=var_name,
                                     var_unit=var_unit, fig_filename=os.path.join(output_path, path),
                                     circular=circular, label_empirical=label_empirical,
                                     label_kernel=label_kernel)

    elements.append([path, kind, caption])

    return pd.DataFrame(elements, columns=['path', 'kind', 'caption']), title
