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

from report.util.template import get_output_name, get_values


def get_info(metocean, section, descriptor):
    info = {}

    for key in metocean.keys():
        value = value_representation(metocean, key)
        info[key + '_metocean'] = value

    info['name_section'] = section.name
    for key in get_values(section):
        value = value_representation(section, key)
        info[key + '_section'] = value

    if descriptor is not None:
        info['name_block'] = section.parent.name
        for key in get_values(section.parent):
            value = value_representation(section.parent, key)
            info[key + '_block'] = value

        info['name_driver'] = descriptor.parent.name
        for key in get_values(descriptor.parent):
            value = value_representation(descriptor.parent, key)
            info[key + '_driver'] = value

        info['name_descriptor'] = descriptor.name
        for key in get_values(descriptor):
            value = value_representation(descriptor, key)
            info[key + '_descriptor'] = value

    return info


def value_representation(section, key):
    value = section[key]

    if not isinstance(value, list):
        try:
            value = section.as_bool(key)
        except ValueError:
            try:
                value = section.as_int(key)
            except ValueError:
                try:
                    value = section.as_float(key)
                except ValueError:
                    pass

    return value


def input_tex(elements, info, output_path, section):
    kind = 'tex'

    driver = info['name_driver']
    location = info['location_metocean']
    descriptor = info['name_descriptor']
    block = info['name_block']

    path = get_output_name(location=location, driver=driver, name=descriptor, block=block, title=section, kind=kind)

    open(os.path.join(output_path, path), 'a').close()

    elements.append([path, kind, None])
