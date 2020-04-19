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
import gettext


def select_language(conf):
    domain = 'messages'
    localedir = os.path.join('..', 'locale')

    if conf['LANGUAGE']['lang'] == 'spanish':
        es = gettext.translation(domain, localedir=localedir, languages=['es'], fallback=True)
        #es.install(unicode=True)
        es.install()
    else:
        default = gettext.translation(domain, localedir=localedir, languages=[], fallback=True)
        #default.install(unicode=True)
        default.install()
