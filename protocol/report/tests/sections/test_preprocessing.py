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

from metoceandataframe.metoceandataframe import MetOceanDF
import report.sections
import report.util
from report import tests


def test_create_report_figures():
    # Input
    modf_file_name = 'guadalete_estuary_wave.modf'
    modf_path = os.path.join(tests.current_path, '..', '..', 'inputadapter',  'tests', 'output', 'modf')
    main_descriptor_name = 'Hm0'
    # File labels
    location = 'guadalete_estuary'
    driver = 'wave'
    block = 'preprocessing'

    # Read data
    path_name = os.path.join(modf_path, modf_file_name)
    modf = MetOceanDF.read_file(path_name)
    # Extract main descriptor
    main_descriptor = report.util.driver.extract_main_descriptor(modf, main_descriptor_name=main_descriptor_name)

    # Create figures of the report
    report.sections.preprocessing.create_report_figures(main_descriptor, location, driver, block, report_type='simple')
