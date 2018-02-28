import decimal
from typing import List, Dict, Set, Any, Optional, Union, Tuple
import numpy as np
from numpy import ndarray
from dexplo._libs import validate_arrays as va

_DT = {'i': 'int', 'f': 'float', 'b': 'bool', 'O': 'str', 'M': 'datetime64[ns]', 'm': 'timedelta64[ns]'}
_DT_GENERIC = {'i': 'int', 'f': 'float', 'b': 'bool', 'O': 'str', 'M': 'date', 'm': 'date'}
_KIND = {'int': 'i', 'float': 'f', 'bool': 'b', 'str': 'O'}
_KIND_LIST = {'int': ['i'], 'float': ['f'], 'bool': ['b'], 'str': ['O'], 'number': ['i', 'f'],
              'datetime': 'M', 'timedelta': 'm'}
_DTYPES = {'int': 'int64', 'float': 'float64', 'bool': 'bool', 'str': 'O',
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
_KIND_NP = {'i': 'int64', 'f': 'float64', 'b': 'bool', 'O': 'O',
            'M': 'datetime64[ns]', 'm': 'timedelta64[ns]' }
_NP_KIND = {'int64': 'i', 'float64': 'f', 'bool': 'b', 'O': 'O', 'U': 'U'}

_AXIS = {'rows': 0, 'columns': 1}
_NON_AGG_FUNCS = {'cumsum', 'cummin', 'cummax', 'cumprod'}
_COLUMN_STACK_FUNCS = {'cumsum', 'cummin', 'cummax', 'mean', 'median', 'var', 'std',
                       'argmax', 'argmin', 'quantile', 'nunique', 'prod', 'cumprod', 'mode'}

_SPECIAL_METHODS = {'__sub__': 'subtraction', '__mul__': 'multiplication',
                    '__pow__': 'exponentiation', '__rsub__': '(right) subtraction'}

ColumnSelection = Union[int, str, slice, List[Union[str, int]]]
RowSelection = Union[int, slice, List[int], 'DataFrame']


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


def get_num_cols(arrs: List[ndarray]) -> int:
    col_length: int = 0
    arr: ndarray

    for arr in arrs:
        col_length += arr.shape[1]
    return col_length


def get_decimal_len(num: float) -> int:
    if not np.isfinite(num):
        return 0
    return abs(decimal.Decimal(str(num)).as_tuple().exponent)


def get_whole_len(num: float) -> int:
        return len(str(num).split('.')[0])


def check_duplicate_list(lst: List[str]) -> None:
    s: Set[str] = set()
    for i, elem in enumerate(lst):
        if elem in s:
            raise ValueError(f'Column {elem} is selected more than once')
        s.add(elem)


def check_empty_slice(s: slice) -> bool:
    return (s.start is None and
            s.stop is None and
            (s.step is None or s.step == 1))


def try_to_squeeze_array(arr: ndarray) -> ndarray:
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2 and (arr.shape[0] == 1 or arr.shape[1] == 1):
        return arr.squeeze()
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
                              np.datetime64, np.timedelta64))


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


def convert_list_to_single_arr(values: List) -> ndarray:
    arr: ndarray = np.array(values)
    kind: str = arr.dtype.kind
    if kind in 'ifbO':
        return arr
    elif kind in 'US':
        return np.array(values, dtype='O')


def maybe_convert_1d_array(arr: ndarray, column: Optional[str]=None) -> ndarray:
    arr = try_to_squeeze_array(arr)
    kind: str = arr.dtype.kind
    if kind in 'ifb':
        return arr
    elif kind == 'M':
        return arr.astype('datetime64[ns]')
    elif kind == 'm':
        return arr.astype('timedelta64[ns]')
    elif kind == 'U':
        return arr.astype('O')
    elif kind == 'O':
        return va.validate_strings_in_object_array(arr, column)
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
    if data.dtype.kind not in 'bifUOMm':
        raise TypeError('Array must be of type boolean, integer, float, string, or unicode')
    if data.ndim == 1:
        return 1
    elif data.ndim == 2:
        return data.shape[1]
    else:
        raise ValueError('Array must be either one or two dimensions')


def is_one_row(num_rows_to_set: int, num_cols_to_set: int) -> bool:
    return num_rows_to_set == 1 and num_cols_to_set >= 1


def convert_list_to_arrays(value: List, single_row: bool) -> List[ndarray]:
    # check if one dimensional array
    arr: ndarray
    if is_scalar(value[0]):
        if single_row:
            arrs = []
            for v in value:
                arr = convert_list_to_single_arr([v])
                arr = convert_list_to_single_arr(arr)
                arrs.append(arr)
            return arrs

        arr = convert_list_to_single_arr(value)
        return [maybe_convert_1d_array(arr)]
    else:
        arr = np.array(value, dtype='O')
        if arr.ndim == 1:
            arr = arr[:, np.newaxis]
    if arr.ndim != 2:
        raise ValueError(f'List converted to {arr.ndim} dimensions. '
                         'Only 1 or 2 dimensions allowed')

    arrs = []
    for i in range(arr.shape[1]):
        a = convert_list_to_single_arr(arr[:, i].tolist())
        a = maybe_convert_1d_array(a)
        arrs.append(a)
    return arrs


def convert_array_to_arrays(arr: ndarray) -> List[ndarray]:
    if arr.ndim == 1:
        arr = arr[:, np.newaxis]
    if arr.ndim != 2:
        raise ValueError('Setting array must be 1 or 2 dimensions')

    arrs: List[ndarray] = []
    i: int
    for i in range(arr.shape[1]):
        a = convert_bytes_or_unicode(arr[:, i])
        if a.dtype.kind == 'O':
            va.validate_strings_in_object_array(a)
        arrs.append(a)
    return arrs


def is_entire_column_selection(rs: Any, cs: Any) -> bool:
    return (isinstance(rs, slice) and isinstance(cs, str) and
            check_empty_slice(rs))


def validate_selection_size(key: Any) -> None:
    if not isinstance(key, tuple):
        raise ValueError('You must provide both a row and column '
                         'selection separated by a comma')
    if len(key) != 2:
        raise ValueError('You must provide exactly one row selection '
                         'and one column selection')


def check_set_value_type(dtype: str, good_dtypes: str, name: str) -> None:
    if dtype not in good_dtypes:
            raise TypeError(f'Cannot assign {name} to column of '
                            f'type {_DT[dtype]}')


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


def convert_dtype_to_kind(dtype: str) -> str:
    return _KIND[dtype]


def get_kind_from_scalar(s: Any) -> str:
    if isinstance(s, bool):
        return 'b'
    elif isinstance(s, (int, np.integer)):
        return 'i'
    elif isinstance(s, (float, np.floating)):
        return 'f'
    elif isinstance(s, (str, bytes)) or s is None:
        return 'O'
    else:
        return ''


def convert_special_method(name):
    return _SPECIAL_METHODS.get(name, 'unknown')


def validate_array_size(arr: ndarray, num_rows: int) -> None:
    if len(arr) != num_rows:
        raise ValueError(f'Mismatch number of rows {len(arr)} vs {num_rows}')


def validate_multiple_string_cols(arr: ndarray) -> ndarray:
    if arr.ndim == 1:
        return va.validate_strings_in_object_array(arr)
    arrays: List[ndarray] = []
    for i in range(arr.shape[1]):
        arrays.append(va.validate_strings_in_object_array(arr[:, i]))
    return np.column_stack(arrays)


def get_selection_object(rs: RowSelection, cs: ColumnSelection) -> Any:
    is_row_list = isinstance(rs, (list, np.ndarray))
    is_col_list = isinstance(cs, (list, np.ndarray))
    if is_row_list and is_col_list:
        return np.ix_(rs, cs)
    return rs, cs


def check_compatible_kinds(kinds1: List[str], kinds2: List[str], all_nans: List[bool]) -> bool:
    for k1, k2, an in zip(kinds1, kinds2, all_nans):
        if k1 == k2:
            continue
        if k1 in 'ifb' and k2 in 'ifb':
            continue
        if k1 in 'O' and an:
            continue
        if k1 in 'mM' and k2 in 'mM':
            continue
        raise TypeError(f'Incompaitble dtypes {_DT[k1]} and {_DT[k2]}')
    return True


def check_all_nans(arrs: List[ndarray]) -> List[bool]:
    all_nans: List[bool] = []
    arr: ndarray

    for arr in arrs:
        if arr.dtype.kind in 'ibO':
            all_nans.append(False)
        else:
            all_nans.append(np.isnan(arr).all())
    return all_nans


def convert_axis_string(axis: str) -> int:
    try:
        return _AXIS[axis]
    except KeyError:
        raise KeyError('axis must be either "rows" or "columns')


def convert_clude(clude: Union[str, List[str]], arg_name: str) -> Union[str, List[str]]:
    all_clude: Union[str, List[str]]
    if isinstance(clude, str):
        all_clude = try_to_convert_dtype(clude)
    elif isinstance(clude, list):
        all_clude = []
        for dt in clude:
            all_clude.extend(try_to_convert_dtype(dt))
    else:
        raise ValueError(f'Must pass a string or list of strings '
                         'to {arg_name}')
    return all_clude


def try_to_convert_dtype(dtype: str) -> List[str]:
    try:
        return _KIND_LIST[dtype]
    except KeyError:
        raise KeyError(f"{dtype} must be one/list of "
                       "either ('float', 'integer', 'bool',"
                       "'str', 'number', 'datetime', 'timedelta')")


def swap_axis_name(axis: str) -> str:
    if axis == 'rows':
        return 'columns'
    if axis == 'columns':
        return 'rows'
    raise ValueError('axis must be either "rows" or "columns"')


def concat_stat_arrays(data_dict: Dict[str, List[ndarray]]) -> Dict[str, ndarray]:
    new_data: Dict[str, ndarray] = {}
    for dtype, arrs in data_dict.items():
        if len(arrs) == 1:
            new_data[dtype] = np.asfortranarray(arrs[0])
        else:
            arrs = np.column_stack(arrs)
            new_data[dtype] = np.asfortranarray(arrs)
    return new_data


def is_agg_func(name: str) -> bool:
    return name not in _NON_AGG_FUNCS


def is_column_stack_func(name: str) -> bool:
    return name in _COLUMN_STACK_FUNCS

