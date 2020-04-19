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
from pylatex import Command, NoEscape, Document, Figure, Table, Tabularx


def document(documentclass='book', document_options='11pt,fleqn', fontenc=None, inputenc=None, lmodern=False,
             textcomp=False, page_numbers=False):
    doc = Document(documentclass=documentclass, document_options=document_options, fontenc=fontenc, inputenc=inputenc,
                   lmodern=lmodern, textcomp=textcomp, page_numbers=page_numbers)

    return doc


def preamble(doc, template):
    # Macros
    doc.preamble.append(Command('input', NoEscape('sty/macros')))

    # Language
    doc.preamble.append(Command('doclanguage', NoEscape(template['LANGUAGE']['lang'])))

    doc.preamble.append(Command('docimages', NoEscape('img/')))

    # Variables
    doc.preamble.append(Command('doctitle', NoEscape(template['VARIABLES']['title'])))
    doc.preamble.append(Command('docsubtitle', NoEscape(template['VARIABLES']['subtitle'])))
    doc.preamble.append(Command('docauthor', NoEscape(template['VARIABLES']['author'])))
    doc.preamble.append(Command('doccopyright', NoEscape(template['VARIABLES']['copyright'])))
    doc.preamble.append(Command('docpublisher', NoEscape(template['VARIABLES']['publisher'])))
    doc.preamble.append(Command('docbookurl', NoEscape(template['VARIABLES']['url'])))
    doc.preamble.append(Command('doclicense', NoEscape(template['VARIABLES']['license'])))
    doc.preamble.append(Command('docedition', NoEscape(template['VARIABLES']['edition'])))

    # Packages
    doc.preamble.append(Command('usepackage', NoEscape('sty/legrand')))
    doc.preamble.append(Command('usepackage', NoEscape('sty/legrand-enhanced')))
    doc.preamble.append(Command('usepackage', NoEscape('sty/custom')))

    # Style
    doc.append(Command('input', NoEscape('sty/title')))
    doc.append(Command('input', NoEscape('sty/copyright')))
    doc.append(Command('input', NoEscape('sty/toc')))


def add_elements(doc, elements):
    for row in elements.itertuples(index=False):
        if row.kind == 'figure':
            add_figure(doc, row.path, row.caption)
        elif row.kind == 'table':
            add_table(doc, row.path, row.caption)
        elif row.kind == 'tex':
            add_tex(doc, row.path)


def add_figure(doc, path, caption):
    with doc.create(Figure()) as figure:
        figure.add_image(filename=path)  # , image_options='width=300px'

        figure.add_caption(caption)


def add_table(doc, path, caption):
    table_data = pd.read_csv(path)

    alignment = 'X'
    n_columns = table_data.shape[1]
    fmt = '{}|{}'.format(alignment, alignment*(n_columns-1))

    with doc.create(Table()) as table:
        with doc.create(Tabularx(fmt)) as tabularx:
            # Table header
            tabularx.add_hline()
            empty_title_columns = table_data.columns.str.contains('unnamed', case=False)
            title_columns = table_data.columns.values
            title_columns[empty_title_columns] = ''
            tabularx.add_row(title_columns)
            tabularx.add_hline()

            # Data
            for row in table_data.itertuples(index=False):
                tabularx.add_row(row)
            tabularx.add_hline()

        table.add_caption(caption)


def add_tex(doc, path):
    path = NoEscape(path)
    path = path.replace('\\', '/')
    doc.append(Command('input', NoEscape(path)))
