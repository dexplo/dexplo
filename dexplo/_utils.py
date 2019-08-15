import decimal
from typing import List, Dict, Set, Any, Union, Tuple
import numpy as np
from numpy import ndarray
from ._libs import validate_arrays as va

_DT = {'i': 'int', 'f': 'float', 'b': 'bool', 'S': 'str',
       'M': 'datetime64[ns]', 'm': 'timedelta64[ns]'}
_DT_GENERIC = {'i': 'int', 'f': 'float', 'b': 'bool', 'S': 'str', 'M': 'date', 'm': 'date'}
_DT_FUNC_NAME = {'i': 'int', 'f': 'float', 'b': 'bool', 'S': 'str',
                 'M': 'datetime', 'm': 'timedelta'}
_DT_STAT_FUNC_NAME = {'i': 'int', 'f': 'float', 'b': 'bool', 'S': 'str', 'M': 'date', 'm': 'date'}

_KIND = {'int': 'i', 'float': 'f', 'bool': 'b', 'str': 'S'}
_KIND_LIST = {'int': ['i'], 'float': ['f'], 'bool': ['b'], 'str': ['S'], 'number': ['i', 'f'],
              'datetime': ['M'], 'timedelta': ['m']}
_DTYPES = {'int': 'int64', 'float': 'float64', 'bool': 'bool', 'str': 'S',
           'datetime64[ns]': 'datetime64[ns]', 'datetime64[us]': 'datetime64[us]',
           'datetime64[ms]': 'datetime64[ms]', 'datetime64[s]': 'datetime64[s]',
           'datetime64[m]': 'datetime64[m]', 'datetime64[h]': 'datetime64[h]',
           'datetime64[D]': 'datetime64[D]', 'datetime64[W]': 'datetime64[W]',
           'datetime64[M]': 'datetime64[M]', 'datetime64[Y]': 'datetime64[Y]',
           'timedelta64[ns]': 'timedelta64[ns]', 'timedelta64[us]': 'timedelta64[us]',
           'timedelta64[ms]': 'timedelta64[ms]', 'timedelta64[s]': 'timedelta64[s]',
           'timedelta64[m]': 'timedelta64[m]', 'timedelta64[h]': 'timedelta64[h]',
           'timedelta64[D]': 'timedelta64[D]', 'timedelta64[W]': 'timedelta64[W]',
           'timedelta64[M]': 'timedelta64[M]', 'timedelta64[Y]': 'timedelta64[Y]'
           }
_KIND_NP = {'i': 'int64', 'f': 'float64', 'b': 'bool', 'S': 'uint32',
            'M': 'datetime64[ns]', 'm': 'timedelta64[ns]'}
_NP_KIND = {'int64': 'i', 'float64': 'f', 'bool': 'b', 'S': 'S', 'U': 'U'}

_AXIS = {'rows': 0, 'columns': 1}
_NON_AGG_FUNCS = {'cumsum', 'cummin', 'cummax', 'cumprod'}
_COLUMN_STACK_FUNCS = {'cumsum', 'cummin', 'cummax', 'mean', 'median', 'var', 'std',
                       'argmax', 'argmin', 'quantile', 'nunique', 'cumprod', 'mode'}

_SPECIAL_METHODS = {'__sub__': 'subtraction', '__mul__': 'multiplication',
                    '__pow__': 'exponentiation', '__rsub__': '(right) subtraction'}

_SPECIAL_OPS = {'__add__': '+', '__radd__': '+', '__mul__': '*', '__rmul__': '*', '__sub__': '-',
                '__rsub__': '-', '__truediv__': '/', '__rtruediv__': '/', '__floordiv__': '//',
                '__rfloordiv__': '//', '__pow__': '**', '__rpow__': '**', '__mod__': '%',
                '__rmod__': '%', '__gt__': '>', '__ge__': '>=', '__lt__': '<', '__le__': '<=',
                '__eq__': '==', '__ne__': '!=', '__neg__': '-'}

# make full mapping from special method to common name for better error messages?

ColumnSelection = Union[int, str, slice, List[Union[str, int]]]
RowSelection = Union[int, slice, List[int], 'DataFrame']

MIN_INT = np.iinfo('int64').min
NaT = np.datetime64('NaT')


class Column:

    def __init__(self, dtype: str = '', loc: int = -1, order: int = -1) -> None:
        self.dtype = dtype
        self.loc = loc
        self.order = order

    @property
    def values(self) -> Tuple[str, int, int]:
        return self.dtype, self.loc, self.order

    def __repr__(self) -> str:
        return f'dtype={self.dtype}, loc={self.loc}, order={self.order}'

    def __eq__(self, other):
        for v1, v2 in zip(self.values, other.values):
            if v1 != v2:
                return False
        return True


def get_num_cols(arrs: List[ndarray]) -> int:
    col_length: int = 0

    for arr in arrs:
        col_length += arr.shape[1]
    return col_length


def get_decimal_len(num: Union[float, str]) -> int:
    if isinstance(num, str):
        return 0
    if not np.isfinite(num):
        return 0
    return abs(decimal.Decimal(str(num)).as_tuple().exponent)


def get_whole_len(num: float) -> int:
        return len(str(num).split('.')[0])


def check_duplicate_column(col_list: List[str]) -> None:
    s: Set[str] = set()
    for col in col_list:
        if col in s:
            raise ValueError(f'Column {col} is selected more than once. You may only select each '
                             f'column once.')
        s.add(col)


def check_empty_slice(s: slice) -> bool:
    return (s.start is None and
            s.stop is None and
            (s.step is None or s.step == 1))


def try_to_squeeze_array(arr: ndarray) -> ndarray:
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2 and (arr.shape[0] == 1 or arr.shape[1] == 1):
        return np.atleast_1d(arr.squeeze())
    else:
        raise ValueError('Array must be one dimensional or two dimensional '
                         'with 1 column')


def convert_bytes_or_unicode(arr: ndarray) -> ndarray:
    if arr.dtype.kind == 'S':
        arr = arr.astype('U').astype('O')
    elif arr.dtype.kind == 'U':
        arr = arr.astype('O')
    return arr


def is_scalar(value: Any) -> bool:
    return isinstance(value, (int, str, float, np.number, bool, bytes,
                              np.datetime64, np.timedelta64)) or value is None


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.number))


def is_integer(value: Any) -> bool:
    return isinstance(value, (int, np.integer))


def is_float(value: Any) -> bool:
    return isinstance(value, (float, np.floating))


def get_overall_dtype(value: Any) -> str:
    if is_number(value):
        return 'number'
    if isinstance(value, str):
        return 'str'
    return 'unknown'


def is_compatible_values(v1: Any, v2: Any) -> str:
    overall_dtype1 = get_overall_dtype(v1)
    overall_dtype2 = get_overall_dtype(v2)
    if overall_dtype1 == 'unknown' or overall_dtype2 == 'unknown':
        raise TypeError(f'Incompaitble data types for {v1} and {v2}')
    if overall_dtype1 != overall_dtype2:
        raise TypeError(f'Value {v1} is a {overall_dtype1} while value {v2} '
                        f'is a {overall_dtype2}. They must be the same.')
    return overall_dtype1


def convert_1d_array(arr: ndarray) -> ndarray:
    arr = try_to_squeeze_array(arr)
    kind: str = arr.dtype.kind
    if kind in 'ifU':
        return arr
    elif kind == 'S':
        return arr.astype('U')
    elif kind == 'M':
        return arr.astype('datetime64[ns]')
    elif kind == 'm':
        return arr.astype('timedelta64[ns]')
    elif kind == 'b':
        return arr.astype('int8')
    else:
        raise NotImplementedError(f'Data type {kind} unknown')


def get_datetime_str(arr: ndarray):
    dt = {0: 'ns', 1: 'us', 2: 'ms', 3: 's', 4: 'D'}
    arr = arr[~np.isnat(arr)].view('int64')
    counts = np.zeros(len(arr), dtype='int64')
    for i, val in enumerate(arr):
        if val == 0:
            counts[i] = 4
            continue
        dec = decimal.Decimal(int(val)).as_tuple()
        ct = 0

        for digit in dec.digits[::-1]:
            if digit == 0:
                ct += 1
            else:
                break

        if ct >= 11:
            counts[i] = 4
        else:
            counts[i] = ct // 3

    return dt[counts.min()]


def get_timedelta_str(arr: ndarray):
    max_val = np.abs(arr[~np.isnat(arr)].view('int64')).max()
    if max_val < 10 ** 3:
        unit = 'ns'
    elif max_val < 10 ** 6:
        unit = 'us'
    elif max_val < 10 ** 9:
        unit = 'ms'
    elif max_val < 60 * 10 ** 9:
        unit = 's'
    elif max_val < 3600 * 10 ** 9:
        unit = 'm'
    elif max_val < 3600 * 24 * 10 ** 9:
        unit = 'h'
    else:
        unit = 'D'

    return unit


def validate_array_type_and_dim(data: ndarray) -> int:
    """
    Called when array is passed to `data` parameter in DataFrame constructor.
    Validates that the array is of a specific type and either 1 or 2 dimensions

    Parameters
    ----------
    data: Array

    Returns
    -------
    The number of columns as an integer
    """
    if data.dtype.kind not in 'bifUSOMm':
        raise TypeError('Array must be of type boolean, integer, float, string, or unicode')
    if data.ndim == 1:
        return 1
    elif data.ndim == 2:
        return data.shape[1]
    else:
        raise ValueError('Array must be either one or two dimensions')


def convert_lists_vertical(lists: List):
    if len(lists) == 0:
        raise ValueError('Cannot set with any empty list')
    if isinstance(lists[0], list):
        new_lists = []
        for i, lst in enumerate(lists):
            for j, val in enumerate(lst):
                if i == 0:
                    new_lists.append([])
                new_lists[j].append(val)
            if i > 0:
                if len(new_lists[0]) != len(new_lists[i]):
                    raise ValueError('You are setting with unequal list sizes. Column 0 has length '
                                     f'{len(new_lists[0])} while column {i} has length '
                                     f'{len(new_lists[i])} ')
        return new_lists
    else:
        return lists


def convert_to_arrays(value: Union[List, ndarray], ncols_to_set: int, cur_kinds: List[str]) -> List[ndarray]:

    if isinstance(value, ndarray):
        if value.ndim == 1:
            pass
        elif value.ndim == 2:
            value = [value[:, i] for i in range(value.shape[1])]
        else:
            raise ValueError('Setting array must be 1 or 2 dimensions')

    arrs: List[ndarray] = []
    kinds: List[str] = []
    srms: List[List] = []
    # Assume each item in the list/array is a column
    if ncols_to_set > 1:
        for val, cur_kind in zip(value, cur_kinds):
            if is_scalar(val):
                val = [val]
            elif not isinstance(val, (list, ndarray)):
                raise TypeError("When setting multiple columns, provide a list of scalars, "
                                f"lists, arrays. You provided a list of {type(val)}")
            if isinstance(val, list):
                result, kind, srm = va.convert_object_array_with_kinds(val, cur_kind)
            elif isinstance(val, ndarray):
                kind = val.dtype.kind
                srm = []
                if kind == 'O':
                    result, kind, srm = va.convert_object_array_with_kinds(val, cur_kind)
                elif kind == 'b':
                    result = val.astype('int8')
                elif kind in 'SU':
                    if kind == 'S':
                        val = val.astype('U')
                    result, kind, srm = va.convert_str_to_cat(val)
                else:
                    result = val

            if check_all_nans(result, kind):
                result = get_missing_value_array(cur_kind, len(val))
                kind = cur_kind
                if kind == 'S':
                    srm = [False]

            arrs.append(result)
            kinds.append(kind)
            srms.append(srm)
    else:
        # setting a single column
        cur_kind = cur_kinds[0]
        if isinstance(value, list):
            result, kind, srm = va.convert_object_array_with_kinds(value, cur_kind)
        elif isinstance(value, ndarray):
            kind = value.dtype.kind
            srm = []
            # check all missing
            if kind == 'O':
                result, kind, srm = va.convert_object_array_with_kinds(value, cur_kind)
            elif kind == 'b':
                result = value.astype('int8')
            elif kind in 'SU':
                if kind == 'S':
                    value = value.astype('U')
                result, kind, srm = va.convert_str_to_cat(value)
            else:
                result = value
            if check_all_nans(result, kind):
                result = get_missing_value_array(cur_kind, len(value))
                kind = cur_kind
                if kind == 'S':
                    srm = [False]

        arrs.append(result)
        kinds.append(kind)
        srms.append(srm)
    return arrs, kinds, srms


def setitem_validate_col_types(cur_kinds: List[str], kinds: List[str], cols: List[str]) -> None:
    """
    Used to verify column dtypes when setting a scalar
    to many columns
    """
    for cur_kind, kind, col in zip(cur_kinds, kinds, cols):
        if cur_kind == kind or (cur_kind in 'if' and kind in 'if') or kind == 'missing':
            continue
        else:
            dt: str = convert_kind_to_dtype(kind)
            ct: str = convert_kind_to_dtype(cur_kind)
            raise TypeError(f'Trying to set a {dt} on column {col} which has type {ct}')


def setitem_validate_scalar_col_types(cur_kinds: List[str], kind: str, cols: List[str]) -> None:
    """
    Used to verify column dtypes when setting a scalar
    to many columns
    """
    for cur_kind, col in zip(cur_kinds, cols):
        if cur_kind == kind or (cur_kind in 'if' and kind in 'if') or kind == 'missing':
            continue
        else:
            dt: str = convert_kind_to_dtype(kind)
            ct: str = convert_kind_to_dtype(cur_kind)
            raise TypeError(f'Trying to set a {dt} on column {col} which has type {ct}')


def setitem_validate_shape(nrows_to_set: int, ncols_to_set: int,
                            other: Union[List, 'DataFrame']) -> None:
    if isinstance(other, list):
        nrows_set = len(other[0])
        ncols_set = len(other)
    # Otherwise we have a DataFrame
    else:
        nrows_set = other.shape[0]
        ncols_set = other.shape[1]

    if nrows_to_set != nrows_set:
        raise ValueError(f'Mismatch of number of rows {nrows_to_set} != {nrows_set}')
    if ncols_to_set != ncols_set:
        raise ValueError(f'Mismatch of number of columns {ncols_to_set} != {ncols_set}')


def is_entire_column_selection(rs: Any, cs: Any) -> bool:
    return (isinstance(rs, slice) and isinstance(cs, str) and
            check_empty_slice(rs))


def validate_selection_size(key: Any) -> None:
    if not isinstance(key, tuple):
        raise ValueError('You must provide both a row and column '
                         'selection separated by a comma')
    if len(key) != 2:
        raise ValueError('You must provide exactly one row selection '
                         'and one column selection separated by a comma. '
                         f'You provided {len(key)} selections.')


def check_valid_dtype_convert(dtype: str) -> str:
    if dtype not in _DTYPES:
        raise ValueError(f'{dtype} is not a valid type. Must be one '
                         'of int, float, bool, str, datetime64[X], timedelta64[X], '
                         'where `X` is one of ns, us, ms, s, m, h, D, W, M, Y')
    return _DTYPES[dtype]


def convert_kind_to_dtype(kind: str) -> str:
    return _DT[kind]


def convert_kind_to_dtype_generic(kind: str) -> str:
    return _DT_GENERIC[kind]


def convert_kind_to_numpy(kind: str) -> str:
    return _KIND_NP[kind]


def convert_numpy_to_kind(dtype: str) -> str:
    try:
        return _NP_KIND[dtype]
    except KeyError:
        dt = dtype.split('[')[0]
        if dt == 'datetime64':
            return 'M'
        elif dt == 'timedelta64':
            return 'm'


def convert_dtype_to_func_name(dtype: str) -> str:
    return _DT_FUNC_NAME[dtype]


def get_stat_func_name(name: str, dtype: str) -> str:
    dtype_name: str = _DT_STAT_FUNC_NAME[dtype]
    return f'{name}_{dtype_name}'


def validate_array_size(arr, num_rows: int) -> None:
    if len(arr) != num_rows:
        raise ValueError(f'Mismatch number of rows {len(arr)} vs {num_rows}')


def check_all_nans(arr: ndarray, kind: str) -> bool:
    if kind == 'b':
        return (arr == -1).all()
    elif kind == 'i':
        return (arr == MIN_INT).all()
    elif kind == 'S':
        return (arr == 0).all()
    elif kind == 'f':
        return np.isnan(arr).all()
    elif kind in 'mM':
        return np.isnat(arr).all()


def convert_axis_string(axis: str) -> int:
    try:
        return _AXIS[axis]
    except KeyError:
        raise KeyError('axis must be either "rows" or "columns')


def try_to_convert_dtype(dtype: str) -> List[str]:
    try:
        return _KIND_LIST[dtype]
    except KeyError:
        raise KeyError(f"{dtype} must be one/list of "
                       "either ('float', 'integer', 'bool',"
                       "'str', 'number', 'datetime', 'timedelta')")


def convert_clude(clude: Union[str, List[str]]) -> List[str]:
    all_clude: List[str]
    if isinstance(clude, str):
        all_clude = try_to_convert_dtype(clude)
    elif isinstance(clude, list):
        all_clude = []
        for dt in clude:
            all_clude.extend(try_to_convert_dtype(dt))
    else:
        raise ValueError(f'Must pass a string or list of strings '
                         'to {arg_name}')
    return set(all_clude)


def swap_axis_name(axis: str) -> str:
    if axis == 'rows':
        return 'columns'
    if axis == 'columns':
        return 'rows'
    raise ValueError('axis must be either "rows" or "columns"')


def create_empty_arrs(data_dict):
    empty_arrs = {}
    for dtype, list_arrs in data_dict.items():
        nc = 0
        for arr in list_arrs:
            if arr.ndim == 1:
                nc += 1
            else:
                nc += arr.shape[1]
        nr = len(list_arrs[0])
        if nc > 1:
            empty_arrs[dtype] = np.empty((nr, nc), _KIND_NP[dtype], 'F')
    return empty_arrs


def get_missing_value_code(kind):
    if kind == 'b':
        return -1
    elif kind == 'i':
        return MIN_INT
    elif kind == 'f':
        return np.nan
    elif kind in 'mM':
        return NaT
    elif kind == 'S':
        return 0


def get_missing_value_array(kind, n):
    if kind == 'b':
        return np.full(n, -1, 'int8', 'F')
    elif kind == 'i':
        return np.full(n, MIN_INT, 'int64', 'F')
    elif kind == 'f':
        return np.full(n, np.nan, 'float64', 'F')
    elif kind == 'M':
        return np.full(n, NaT, 'datetime64[ns]', 'F')
    elif kind == 'M':
        return np.full(n, NaT, 'timedelta64[ns]', 'F')
    elif kind == 'S':
        return np.full(n, 0, 'int32', 'F')


def isna_array(arr, dtype):
    if dtype == 'b':
        return arr == -1
    elif dtype == 'i':
        return arr == MIN_INT
    elif dtype == 'f':
        return np.isnan(arr)
    elif dtype in 'mM':
        return np.isnat(arr)
    elif dtype == 'S':
        return arr == 0


def concat_stat_arrays(data_dict: Dict[str, List[ndarray]]) -> Dict[str, ndarray]:
    new_data: Dict[str, ndarray] = {}
    empty_arrs = create_empty_arrs(data_dict)
    for dtype, arrs in data_dict.items():
        if len(arrs) == 1:
            arr = arrs[0]
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            new_data[dtype] = np.asfortranarray(arr)
        else:
            data = empty_arrs[dtype]
            i = 0
            for arr in arrs:
                if arr.ndim == 1:
                    data[:, i] = arr
                    i += 1
                else:
                    for j in range(arr.shape[1]):
                        data[:, i] = arr[:, j]
                        i += 1
            new_data[dtype] = data
    return new_data


def is_agg_func(name: str) -> bool:
    return name not in _NON_AGG_FUNCS


def is_column_stack_func(name: str) -> bool:
    return name in _COLUMN_STACK_FUNCS


def validate_agg_func(name, dtype):
    if dtype in 'ibfm':
        if name not in {'size', 'count', 'sum', 'prod', 'mean',
                        'max', 'min', 'first', 'last', 'var',
                        'cov', 'corr', 'any', 'all', 'median',
                        'nunique'}:
            dtype = convert_kind_to_dtype(dtype)
            raise ValueError(f'Function name {name} does not work for columns of type {dtype}')
    elif dtype in 'OM':
        if name not in {'size', 'count', 'max', 'min', 'first',
                        'last', 'any', 'all', 'nunique'}:
            dtype = convert_kind_to_dtype(dtype)
            raise ValueError(f'Function name {name} does not work for columns of type {dtype}')



