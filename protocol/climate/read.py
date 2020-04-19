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
import os
import re


def simar(file_name, path='.', null_values=(-99.9, -99.99, -999, -9999, 990), columns='std'):
    """Read SIMAR file

    Args:
        file_name (str): file name
        path (str, optional): path
        null_values (list, optional): considered null values
        columns (str, optional):

            * std: Hm0 (significant wave height), Tp (peak period), DirM (mean wave direction),
              VelV (wind velocity) and DirV (mean wind direction)
            * all: all the variables

    Returns:
        tuple:

            * DataFrame: climate agents variables
            * str: identifier of SIMAR file
    """
    n_header_lines, has_header, code = simar_header(os.path.join(path, file_name))

    data = pd.read_table(os.path.join(path, file_name), delim_whitespace=True, skiprows=n_header_lines,
                         parse_dates=[[0, 1, 2, 3]], index_col=0, header=has_header,
                         na_values=null_values)

    if columns == 'std':
        data = data.drop(['Tm02', 'Hm0_V', 'DirM_V', 'Hm0_F1', 'Tm02_F1', 'DirM_F1',
                          'Hm0_F2', 'Tm02_F2', 'DirM_F2'], 1)

    return data, code


def simar_header(file_name):
    """Detecting if file has header. If positive, extract number of lines and identifier code

    Args:
        file_name (str): file name

    Returns:
        tuple:

            * int: number of header lines
            * int: data header line
            * str: identifier of SIMAR file
    """
    n_lines = -1
    code = -1
    has_header = None

    code_pattern = r'CODIGO\D*(\d+)'
    data_pattern = r'\d{4}\s*(\d{1,2}\s*){3,4}\s*((-)?\d*(\.\d+)*\s*)*'

    with open(file_name) as f:
        lines = f.readlines()

    for n_line, line in enumerate(lines):
        match_code = re.search(code_pattern, line)
        match_data = re.search(data_pattern, line)
        if match_code:
            code = match_code.group(1)
        elif match_data:
            if n_line > 0:
                while not lines[n_line-1].strip():
                    n_line -= 1
                n_lines = n_line - 1
                has_header = 0
            else:
                n_lines = n_line
            break

    return n_lines, has_header, code
