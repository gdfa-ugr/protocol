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

from configobj import ConfigObj
from validate import Validator


def read(file_name, output):
    # TODO add validation
    # configspec_file = os.path.join('..', 'templates', output, 'configspec')

    config = ConfigObj(file_name, encoding='utf8')  # configspec=configspec_file
    # validator = Validator()
    # results = config.validate(validator)

    # config_data = None
    # if results is True:
    config_data = config

    return config_data


def get_key(element, key, default):
    value = default
    if key in element:
        value = element[key]

    return value


def get_values(element):
    return element.scalars


def get_children(element):
    return element.sections


def get_output_name(block, name, driver, location, title, kind):
    folder_name = kind + 's'
    extensions = {'figure': '.png',
                  'table': '.csv',
                  'tex': '.tex'}

    path_name = os.path.join(location.lower(), driver.lower())

    file_name = '{}-{}-{}{}'.format(name.lower(), block.lower(), title.lower(), extensions[kind])

    path = os.path.join(folder_name, path_name, file_name)

    return path


def check_visibility(info):
    visibility = True
    if info.get('visible_section') is False:
        visibility = False
    elif info.get('visible_block') is False:
        visibility = False
    elif 'ignore_sections_descriptor' in info and info['name_section'] in info['ignore_sections_descriptor']:
        visibility = False

    return visibility


def save_table(data, path):
    data.to_csv(path, header=True)
