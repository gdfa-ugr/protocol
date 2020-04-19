#!/usr/bin/env python
# coding: utf-8
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

try:
    # noinspection PyUnresolvedReferences, PyCompatibility
    from builtins import *  # noqa
except ImportError:
    pass

import pandas as pd
import scipy.stats as st


def initial_params(data):
    x_lim = data.median()

    x_1, x_2 = data[data <= x_lim], data[data > x_lim]
    alpha = len(x_1) / len(data)

    mu_1, mu_2, sigma_1, sigma_2 = x_1.mean(), x_2.mean(), x_1.std(), x_2.std()

    return alpha, mu_1, sigma_1, mu_2, sigma_2


def norm_norm(data, params, cumulative=False):
    mu_1, mu_2, sigma_1, sigma_2, thold = get_params(params)

    data.sort_values(inplace=True)

    if cumulative:
        norm = getattr(st.norm, 'cdf')
    else:
        norm = getattr(st.norm, 'pdf')

    dist = (thold * norm(data, mu_1, sigma_1) +
            (1 - thold) * norm(data, mu_2, sigma_2))

    return pd.Series(dist, index=data)


def pdf(data, params):
    return norm_norm(data, params)


def cdf(data, params):
    return norm_norm(data, params, cumulative=True)


def get_params(params):
    thold = params[0]
    mu_1 = params[1]
    sigma_1 = params[2]
    mu_2 = params[3]
    sigma_2 = params[4]

    return mu_1, mu_2, sigma_1, sigma_2, thold