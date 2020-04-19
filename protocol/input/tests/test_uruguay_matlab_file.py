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
import pytest

from input import uruguay_matlab_file, tests


def test_uruguay_wave_time_series_file():
    name = 'Oleaje.mat'
    data_path = os.path.join(tests.full_data_path, 'matlab', name)
    data = uruguay_matlab_file.read_file(data_path, variables='JL1cor', time_column=0)

    data.columns = ['t', 'Hs', 'Tp', 'DirM']

    tolerance = 0.01
    assert data.loc['1984-07-04 12', 'Hs'] == pytest.approx(0.2730, tolerance)
    assert data.loc['1997-12-01 09', 'Tp'] == pytest.approx(2.4269, tolerance)
    assert data.loc['2010-12-31 21', 'DirM'] == pytest.approx(145.9055, tolerance)

    # TODO Añadir tolerancia en el resto de test
