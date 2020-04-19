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
from importlib import import_module

from pylatex import NoEscape
from pylatex.section import Part, Chapter, Section, Subsection

from report.sections.common import get_info
from report.sections import select_language
from report.util.template import read, get_children, get_key, check_visibility
from report.sections.latex import preamble, document, add_elements


def create_document(modf, template, output_path='output', output_title='document'):
    conf = read(template, 'latex')
    select_language(conf)

    doc = document()
    preamble(doc, conf)

    has_parts = False
    if isinstance(modf, list):
        has_parts = True
        for mo_info in modf:
            create_parts(conf, doc, mo_info, has_parts, output_path)
    else:
        create_parts(conf, doc, modf, has_parts, output_path)

    doc.generate_pdf(os.path.join(output_path, output_title), clean_tex=False)


def create_parts(conf, doc, modf, has_parts, output_path):
    metocean = conf['METOCEAN']
    driver = modf.driver.upper()

    if has_parts:
        with doc.create(Part(conf['DRIVERS'][driver]['title'])):
            create_part(conf, doc, driver, metocean, modf, output_path)
    else:
        create_part(conf, doc, driver, metocean, modf, output_path)


def create_part(conf, doc, driver, metocean, modf, output_path):
    info_driver = conf['DRIVERS'][driver]
    sections = conf['SECTIONS']
    for section in get_children(sections):
        info_section = sections[section]

        info = get_info(metocean, info_section, None)
        if check_visibility(info) is True:
            with doc.create(Chapter(NoEscape(get_key(info_section, 'title', section)))):
                for descriptor in get_children(info_driver):
                    info_descriptor = info_driver[descriptor]

                    info = get_info(metocean, info_section, info_descriptor)
                    if check_visibility(info) is True:
                        with doc.create(Section(NoEscape(get_key(info_descriptor, 'title', descriptor)))):
                            for subsection in get_children(info_section):
                                info_subsection = info_section[subsection]

                                module = import_module('.' + section.lower(), 'report.sections')
                                func = 'output_' + subsection.lower()

                                info = get_info(metocean, info_subsection, info_descriptor)
                                if check_visibility(info) is True:
                                    call = getattr(module, func)
                                    elements, title = call(modf, info, output_path)

                                    with doc.create(Subsection(NoEscape(get_key(info_subsection, 'title',
                                                                                   title)))):
                                        add_elements(doc, elements)
