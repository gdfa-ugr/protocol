#!/usr/bin/env python
# coding: utf-8
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

try:
    # noinspection PyUnresolvedReferences, PyCompatibility
    from builtins import *  # noqa
except ImportError:
    pass

# TODO Eliminar importación de num2date (función lenta)
from netCDF4 import num2date
from netCDF4 import Dataset

import numpy as np
import pandas as pd
import os


def read_netcdf_file(data_path, name_descriptor = 'Slp', path='.'):
    dataset = Dataset(os.path.join(path, data_path), 'r')

    # Read the data inside the netcdf file
    msl = dataset['msl'][:] / 100  # It is a matrix with four values of msl per time step
    time = dataset['time'][:]
    lat = dataset['latitude'][:]
    lon = dataset['longitude'][:]
    msl_unit = dataset['msl'].units
    lat_unit = dataset['latitude'].units
    lon_unit = dataset['longitude'].units
    time_unit = dataset['time'].units

    # Group in dictionary
    units = {'msl': msl_unit, 'lat': lat_unit, 'lon': lon_unit, 'time': time_unit}
    coordinates = {'lat': lat, 'lon': lon}

    # Read the datetime
    t_cal = dataset.variables['time'].calendar
    datevar = list()
    datevar.append(num2date(time, units=time_unit, calendar=t_cal))
    datevar = datevar[0]

    # Extraction of the datetime index
    indice = pd.Index(datevar)
    msl_vec = np.mean(msl, axis=(1, 2))  # mean value of the four msl values per each time step
    msl_vec = pd.Series(data=msl_vec, index=indice)

    # Add name to Series
    msl_vec.name = name_descriptor

    return msl_vec, units, coordinates


def combination_era_files(msl_vec_inter, msl_vec_40):
    msl_vec = msl_vec_inter.combine_first(msl_vec_40)

    return msl_vec
