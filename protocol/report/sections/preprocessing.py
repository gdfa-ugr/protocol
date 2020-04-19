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

from preprocessing import missing_values
from report.sections.common import input_tex
from report.util.template import get_output_name, save_table, get_key
from report.util.driver import extract_data


def output_missing_values(modf, info, output_path):
    elements = []

    # Section title
    default_title = _('Missing values')

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

    time_step = missing_values.find_timestep(data)
    data_gaps = missing_values.find_missing_values(data, time_step)

    # Figure
    kind = 'figure'
    default_caption = _('Missing values plot') + ': {}'.format(info['title_descriptor'])
    caption = get_key(info, 'missing_values_figure_caption_section', default_caption)

    path = get_output_name(location=location, driver=driver, name=descriptor, block=block, title=section, kind=kind)

    missing_values.plot_missing_values(data=data, data_gaps=data_gaps,
                                       title='', var_name=var_name, var_unit=var_unit,
                                       fig_filename=os.path.join(output_path, path),
                                       circular=circular, label=var_name)

    elements.append([path, kind, caption])

    # Table
    kind = 'table'
    default_caption = _('Missing values table') + ': {}'.format(info['title_descriptor'])
    caption = get_key(info, 'missing_values_table_caption_section', default_caption)

    path = get_output_name(location=location, driver=driver, name=descriptor, block=block, title=section, kind=kind)

    data_gaps_report = missing_values.missing_values_report(data, data_gaps)
    save_table(data_gaps_report, os.path.join(output_path, path))

    elements.append([os.path.join(output_path, path), kind, caption])

    return pd.DataFrame(elements, columns=['path', 'kind', 'caption']), title
