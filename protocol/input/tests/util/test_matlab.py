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

from input import tests
from input.util import matlab


def test_read_time_series_matlab_struct():
    name = 'Oleaje.mat'

    data_path = os.path.join(tests.full_data_path, 'matlab', name)
    data = matlab.read_time_series_matlab(data_path, 'JL1cor', time_column_pos=0, path=data_path)

    print(data)


def test_read_time_series_matlab_no_struct():
    name = 'prueba_datos_matlab.mat'

    data_path = os.path.join(tests.full_data_path, 'matlab', name)

    tolerance = 0.01

    data_1 = matlab.read_time_series_matlab(data_path, 'hs', time_var='t', path=data_path)
    assert data_1.loc['1980-01-02 00'] == pytest.approx(0.4561, tolerance)
    assert data_1.loc['2000-02-12 09'] == pytest.approx(0.2892, tolerance)
    assert data_1.loc['2010-05-05 12'] == pytest.approx(0.1883, tolerance)

    data_2 = matlab.read_time_series_matlab(data_path, 'datos', time_column_pos=0, path=data_path)
    assert data_2.loc['1989-11-22 12', 1] == pytest.approx(0.2914, tolerance)
    assert data_2.loc['1999-02-16 00', 2] == pytest.approx(2.4610, tolerance)
    assert data_2.loc['2009-06-18 03', 3] == pytest.approx(128.7126, tolerance)

    data_3 = matlab.read_time_series_matlab(data_path, ['hs', 'tp', 'dm'], time_var='t', path=data_path)
    assert data_3[0].loc['1980-12-27 15'] == pytest.approx(0.2750, tolerance)
    assert data_3[1].loc['1992-07-13 06'] == pytest.approx(2.5361, tolerance)
    assert data_3[2].loc['2008-02-21 06'] == pytest.approx(130.5889, tolerance)
