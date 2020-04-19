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

import scipy.stats as st
import numpy as np
import matplotlib.pyplot as plt

from climate import tests
from climate.stats import empirical_distributions, fitting
from climate.stats import extremal


def test_plot_empirical_pdf():
    data_simar = tests.read_sample_simar()
    data_column = 'Hm0'

    data = empirical_distributions.epdf_histogram(data_simar[data_column])

    empirical_distributions.plot_empirical(data)


def test_plot_empirical_cdf():
    data_simar = tests.read_sample_simar()
    data_column = 'Hm0'

    data = empirical_distributions.ecdf_histogram(data_simar[data_column])

    empirical_distributions.plot_empirical(data)


def test_ecdf_sm():
    # Read preprocessed SIMAR
    data_path = os.path.join(tests.full_data_path, 'intermediate_files')
    data = tests.read_sample_preprocessed_simar(data_path=data_path)
    hs = data.loc[:, 'Hm0']
    # Calculation of the annual maxima sample
    annual_maxima = extremal.annual_maxima_calculation(hs)
    # Empirical cumulative distribution function
    ecdf_data = empirical_distributions.ecdf_sm(annual_maxima)

    # Plot graphic
    file_name = os.path.join(tests.current_path, 'output', 'img', 'ECDF_annual_maxima_plot')
    empirical_distributions.plot_ecdf_sm_dots(ecdf_data, title='', var_name='Hs', var_unit='m',
                                              fig_filename=file_name, label='ECDF_Annual_Maxima')


def test_plot_empirical_fit():
    data_simar = tests.read_full_simar()
    data_column = 'Hm0'
    fitting_function = st.norm

    data = empirical_distributions.ecdf_histogram(data_simar[data_column])
    _, _, params = fitting.fit(data_simar, data_column, fitting_function)

    empirical_distributions.plot_empirical_fit(data, fitting_function, params, cumulative=True)


def test_plot_kde_scipy():
    data_simar = tests.read_full_simar()
    data_column = 'Hm0'
    cumulative = False
    paso_datos = 0.1
    bins = np.max(data_simar[data_column])/(paso_datos*2.0)

    if cumulative:
        data_empirical = empirical_distributions.ecdf_histogram(data_simar[data_column])
    else:
        data_empirical = empirical_distributions.epdf_histogram(data_simar[data_column], bins=bins)

    data_scipy = empirical_distributions.kde_scipy(data_simar[data_column])
    empirical_distributions.plot_kde(data_empirical, data_scipy, cumulative, title='', var_name=data_column,
                                     var_unit='m', fig_filename='', circular=False, label_empirical='Empirical data',
                                     label_kernel='Empirical kernel fit')


def test_plot_kde_sm():
    data_simar = tests.read_full_simar()
    data_column = 'Hm0'
    cumulative = True

    if cumulative:
        data_empirical = empirical_distributions.ecdf_histogram(data_simar[data_column])
        data_kernel = empirical_distributions.kde_sm(data_simar[data_column], cumulative=cumulative, gridsize=100)
    else:
        data_empirical = empirical_distributions.epdf_histogram(data_simar[data_column])
        data_kernel = None

    empirical_distributions.plot_kde(data_empirical, data_kernel, cumulative, title='', var_name=data_column,
                                     var_unit='m', fig_filename='', circular=False, label_empirical='Empirical data',
                                     label_kernel='Empirical kernel fit')
