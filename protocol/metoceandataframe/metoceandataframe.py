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
import pandas as pd
from pandas.io import packers
from future.utils import bytes_to_native_str as n  # to specify str type on both Python 2/3


class MetOceanDF(pd.DataFrame):
    _metadata = ['source', 'id', 'driver', 'latitude', 'longitude', 'depth']

    @property
    def _constructor(self):
        return MetOceanDF

    def __init__(self, data=None, metadata=None, index=None, columns=None, dtype=None,
                 copy=False):
        if metadata is None:
            metadata = {}

        # noinspection PyArgumentList
        super(MetOceanDF, self).__init__(data=data,
                                         index=index,
                                         columns=columns,
                                         dtype=dtype,
                                         copy=copy)
        for name in self._metadata:
            setattr(self, name, metadata.get(name))

    def get_properties(self):
        properties = {}
        for field in self._metadata:
            properties[field] = getattr(self, field, None)

        return properties

    def get_dataframe(self):
        return pd.DataFrame(self)

    def to_file(self, file_name, path='.'):
        path_name = os.path.join(path, file_name)

        metadata = self.get_properties()

        packers.to_msgpack(path_name, metadata, encoding='utf-8')
        packers.to_msgpack(path_name, self.get_dataframe(), encoding='utf-8', compress=n(b'zlib'), append=True)

    @staticmethod
    def read_file(file_name, path='.'):
        path_name = os.path.join(path, file_name)

        msgpack_data = packers.read_msgpack(path_name, encoding='utf-8')

        metadata = msgpack_data[0]
        df = msgpack_data[1]

        modf = MetOceanDF(df, metadata)

        return modf

    # Prevent custom metadata lost in Pandas (issue 2485):
    # - __finalize__ should work properly
    #       * workaround: override __finalize__ (GeoPandas/xarray approaches)
    # - it is necessary to call __finalize__ in methods (issue 6927)
    #       * workaround: override non-calling methods (i.e. combine_first)

    # __finalize__ GeoPandas implementation
    # noinspection PyCallByClass,PyTypeChecker
    def __finalize__(self, other, method=None, **kwargs):
        """Propagate metadata from other to self """
        # merge operation: using metadata of the left object
        if method == 'merge':
            for name in self._metadata:
                object.__setattr__(self, name, getattr(other.left, name, None))
        # concat operation: using metadata of the first object
        elif method == 'concat':
            for name in self._metadata:
                object.__setattr__(self, name, getattr(other.objs[0], name, None))
        else:
            for name in self._metadata:
                object.__setattr__(self, name, getattr(other, name, None))
        return self

    def combine_first(self, other, *args, **kwargs):
        return super(MetOceanDF, self).combine_first(other, *args, **kwargs).__finalize__(self)
