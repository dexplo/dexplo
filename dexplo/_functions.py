from collections import defaultdict
import numpy as np
from numpy import nan, ndarray
from typing import (Union, Dict, List, Optional, Tuple, Callable, overload,
                    NoReturn, Set, Iterable, Any, TypeVar, Type, Generator)

import dexplo._utils as utils
from dexplo._libs import (read_files as _rf)
from dexplo._frame import  DataFrame


def read_csv(fp, delimitter=','):
    a_bool, a_int, a_float, a_str, columns, dtypes, dtype_loc = _rf.read_csv(fp)

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