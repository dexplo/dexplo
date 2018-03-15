from collections import defaultdict
import numpy as np
from numpy import nan, ndarray
from typing import (Union, Dict, List, Optional, Tuple, Callable, overload,
                    NoReturn, Set, Iterable, Any, TypeVar, Type, Generator)

import dexplo._utils as utils
from dexplo._libs import (read_files as _rf)
from dexplo._frame import DataFrame


def read_csv(fp, sep=',', header=0, skiprows=None):
    if not isinstance(sep, str):
        raise TypeError('`sep` must be a string')
    if len(sep) != 1:
        raise ValueError('`sep` must only be one character in length')
    if not isinstance(header, int):
        raise TypeError('`header` must be an integer')
    if header < -1:
        raise ValueError('`header` must be greater than or equal to -1')

    skiprows_set = set()
    skiprows_int = 0
    if skiprows is None:
        pass
    elif isinstance(skiprows, int):
        if skiprows < 0:
            raise ValueError('`skiprows` must be one or more non-negative integers')
        skiprows_int = skiprows
    else:
        skiprows_arr = np.asarray(skiprows)
        if (skiprows_arr < 0).any():
            raise ValueError('All values in the `skiprows` sequence must be >= 0')
        if header == -1:
            skiprows_set = set(skiprows_arr)
        else:
            max_row = skiprows_arr.max()
            if header > max_row - len(skiprows_arr):
                header += len(skiprows_arr)
            else:
                max_rows = np.arange(max_row)
                kept_rows = max_rows[~np.isin(max_rows, skiprows_arr)]
                header = kept_rows[header]
                skiprows_set = set(skiprows_arr[skiprows_arr > header])

    print(skiprows_int, skiprows_set)

    tuple_return = _rf.read_csv(fp, ord(sep), header, skiprows_int, skiprows_set)

    a_bool, a_int, a_float, a_str, columns, dtypes, dtype_loc = tuple_return

    new_column_info = {}
    dtype_map = {1: 'b', 2: 'i', 3: 'f', 4: 'O'}
    final_dtype_locs = defaultdict(list)
    for i, (col, dtype, loc) in enumerate(zip(columns, dtypes, dtype_loc)):
        new_column_info[col] = utils.Column(dtype_map[dtype], loc, i)
        final_dtype_locs[dtype_map[dtype]].append(loc)

    new_data = {}
    loc_order_changed = set()
    for arr, dtype in zip((a_bool, a_int, a_float, a_str), ('b', 'i', 'f', 'O')):
        num_cols = arr.shape[1]
        if num_cols != 0:
            locs = final_dtype_locs[dtype]
            if len(locs) == num_cols:
                new_data[dtype] = arr
            else:
                loc_order_changed.add(dtype)
                new_data[dtype] = arr[:, locs]

    if loc_order_changed:
        cur_dtype_loc = defaultdict(int)
        for col in columns:
            dtype, loc, order = new_column_info[col].values
            if dtype in loc_order_changed:
                new_column_info[col].loc = cur_dtype_loc[dtype]
                cur_dtype_loc[dtype] += 1

    return DataFrame._construct_from_new(new_data, new_column_info, columns)