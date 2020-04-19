#!/usr/bin/env python
# coding: utf-8
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

try:
    # noinspection PyUnresolvedReferences, PyCompatibility
    from builtins import *  # noqa
except ImportError:
    pass


def extract_data(modf, descriptor_name=None):
    if descriptor_name is None:
        descriptor = modf.iloc[:, 0]
    else:
        descriptor = modf.loc[:, descriptor_name]

    return descriptor
