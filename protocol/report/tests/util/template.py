#!/usr/bin/env python
# coding: utf-8
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

try:
    # noinspection PyUnresolvedReferences, PyCompatibility
    from builtins import *  # noqa
except ImportError:
    pass

import report.util


def test_create_file_name():
    location = 'guadalete_estuary'
    driver = 'wave'
    descriptor = 'hm0'
    block = 'preprocessing'
    figure_name = 'missing_values'
    extension = '.png'

    file_name = report.util.template.get_file_name(location, driver, descriptor, block, figure_name, extension)
