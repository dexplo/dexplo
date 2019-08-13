from collections import defaultdict, OrderedDict
from math import ceil
from copy import deepcopy
from typing import (Union, Dict, List, Optional, Tuple, Callable, overload,
                    NoReturn, Set, Iterable, Any, TypeVar, Type, Generator)
import warnings

import numpy as np
from numpy import nan, ndarray

from . import _init_funcs as init
from . import _utils as utils
from . import options
from ._date import DateTimeClass, TimeDeltaClass
from ._libs import (groupby as _gb,
                    validate_arrays as _va,
                    math as _math,
                    math_oper_string as _mos,
                    sort_rank as _sr,
                    unique as _uq,
                    replace as _repl,
                    pivot as _pivot,
                    out_files as _of,
                    join as _join)
from ._strings import StringClass
from . import _stat_funcs as stat
from ._arithmetic_ops import OP_2D

DataC = Union[Dict[str, Union[ndarray, List]], ndarray]
DictListArr = Dict[str, List[ndarray]]
StrMap = Dict[int, Dict[int, str]]
StrRevMap = Dict[int, List[str]]

# can change to array of strings?
ColumnT = Optional[Union[List[str], ndarray]]
ColInfoT = Dict[str, utils.Column]

IntStr = TypeVar('IntStr', int, str)
IntNone = TypeVar('IntNone', int, None)
ListIntNone = List[IntNone]

ScalarT = TypeVar('ScalarT', int, float, str, bool)

ColSel = Union[int, str, slice, List[IntStr], 'DataFrame']
RowSel = Union[int, slice, List[int], 'DataFrame']
Scalar = Union[int, str, bool, float]
NaT = np.datetime64('nat')
MIN_INT = np.iinfo('int64').min


class DataFrame(object):
    """
    The DataFrame is the primary data container in Dexplo.
        - It consists of exactly two dimensions.
        - Column names must be strings
        - Column names must be unique
        - Each column must be one specific data type
            * integer
            * float
            * boolean
            * str
            * datetime
            * timedelta
    Initializes DataFrame in two specific ways.
    `data` can either be one of the followindg
        - a dictionary of lists or 1d arrays
        - an array

    Parameters
    ----------
    data : Dictionary of lists/1d arrays or an array
    columns : List or array of strings

    Examples
    --------
    >>> df = dx.DataFrame({'State': ['TX', 'FL', 'CO'],
                           'Pop': [22, 19, 7],
                           'Has Bears': [False, True, True]})
    >>> df
       State  Pop  Has Bears
    0     TX   22      False
    1     FL   19       True
    2     CO    7       True

    Subset selection takes place through the [ ].
    You must provide both the row and column selection
    Examples
    >>> df[:, 'Pop']  # or `df[:, 1]` Select an entire column by column name or integer location
       Pop
    0   22
    1   19
    2    7

    >>> df[0, :] # Select an entire row only by integer location
       State  Pop  Has Bears
    0     TX   22      False

    >>> df[[0, -1], ['State', 'Has Bears']] # Use lists to select multiple rows/columns
       State  Has Bears
    0     TX      False
    1     CO       True

    >>> df[:2, [-1, 'Pop']] # Use slice notation or both integers and column names
       Has Bears  Pop
    0      False   22
    1       True   19

    >>> df[df[:, 'Pop'] > 10, 'State':'Pop'] # Use boolean selection
       State  Pop
    0     TX   22
    1     FL   19
    """

    def __init__(self, data: DataC, columns: ColumnT = None) -> None:
        self._columns: ndarray
        self._data: Dict[str, ndarray] = {}
        self._column_info: ColInfoT = {}
        self._hasnans: DictListArr = {}

        if isinstance(data, dict):
            self._columns = init.columns_from_dict(columns, data)
            self._data, self._column_info, self._str_reverse_map = init.data_from_dict(data)

        elif isinstance(data, ndarray):
            num_cols: int = utils.validate_array_type_and_dim(data)
            self._columns = init.columns_from_array(columns, num_cols)
            self._data, self._column_info, self._str_reverse_map = init.data_from_array(data, self._columns)
        else:
            raise TypeError('`data` must be either a dict of arrays or an array')

    @property
    def columns(self) -> List[str]:
        """
        Returns the column names as a list
        """
        return self._columns.tolist()

    # Only gets called after construction, when renaming columns
    @columns.setter
    def columns(self, new_columns: ColumnT) -> None:
        """
        Set column with either list or array with the same number of elements as the columns.
        Each column name must be a string, cannot be duplicated.

        Parameters
        ----------
        new_columns : list or array of unique strings the same length as the current
            number of columns
        """
        new_columns2: ndarray = init.check_column_validity(new_columns)
        len_new: int = len(new_columns2)
        len_old: int = len(self._columns)
        if len_new != len_old:
            raise ValueError(f'There are {len_old} columns in the DataFrame. '
                             f'You provided {len_new}.')

        new_column_info: ColInfoT = {}
        for old_col, new_col in zip(self._columns, new_columns2):
            new_column_info[new_col] = utils.Column(*self._column_info[old_col].values)

        self._column_info = new_column_info
        self._columns = new_columns2

    def _get_col_dtype_loc(self, col):
        col_info = self._column_info[col]
        return col_info.dtype, col_info.loc

    def _get_col_dtype_loc_order(self, col):
        return self._column_info[col].values

    def _col_info_iter(self, with_order=False, with_arr=False):
        for col in self._columns:
            dtype, loc, order = self._get_col_dtype_loc_order(col)
            if with_arr and with_order:
                arr = self._data[dtype][:, loc]
                yield col, dtype, loc, order, arr
            elif with_arr:
                arr = self._data[dtype][:, loc]
                yield col, dtype, loc, arr
            elif with_order:
                yield col, dtype, loc, order
            else:
                yield col, dtype, loc

    @property
    def values(self) -> ndarray:
        """
        Retrieve a single 2-d array of all the data in the correct column order
        """
        if len(self._data) == 1:
            kind: str = next(iter(self._data))
            order: List[int] = [self._column_info[col].loc for col in self._columns]
            arr = self._data[kind][:, order]
            if kind == 'b':
                return arr == 1
            else:
                return arr

        if {'b', 'S', 'm', 'M'} & self._data.keys():
            arr_dtype: str = 'O'
        else:
            arr_dtype = 'float64'

        v: ndarray = np.empty(self.shape, dtype=arr_dtype, order='F')

        for col, dtype, loc, order, col_arr in self._col_info_iter(with_order=True, with_arr=True):
            if dtype == 'S':
                cur_list_map = self._str_reverse_map[loc]
                _va.make_object_str_array(cur_list_map, v, col_arr, order)
            elif dtype == 'M':
                unit = col_arr.dtype.name.replace(']', '').split('[')[1]
                # changes array in place
                _va.make_object_datetime_array(v, col_arr.view('uint64'), order, unit)
            elif dtype == 'm':
                unit = col_arr.dtype.name.replace(']', '').split('[')[1]
                _va.make_object_timedelta_array(v, col_arr.view('uint64'), order, unit)
            else:
                v[:, order] = col_arr
        return v

    def _values_number(self) -> ndarray:
        """
        Retrieve the array that consists only of integer and floats
        Cov and Corr use this
        """
        if 'f' in self._data:
            arr_dtype = 'float64'
        else:
            arr_dtype = 'int64'

        col_num: int = 0
        for kind, arr in self._data.items():
            if kind in 'ifb':
                col_num += arr.shape[1]
        shape: Tuple[int, int] = (len(self), col_num)

        v: ndarray = np.empty(shape, dtype=arr_dtype, order='F')
        for i, (_, col_arr, dtype, _) in enumerate(self._col_info_iter(with_arr=True)):
            if dtype in 'ifb':
                v[:, i] = col_arr
        return v

    def _get_column_values(self, col: str) -> ndarray:
        """
        Retrieve a 1d array of a single column
        """
        dtype, loc = self._get_col_dtype_loc(col)  # type: str, int
        return self._data[dtype][:, loc]

    def _build_repr(self) -> Tuple[List[List[str]], List[int], List[int], List[int]]:
        columns: List[str] = self.columns
        num_rows: int = len(self)
        if len(columns) > options.max_cols:
            col_num: int = options.max_cols // 2
            columns = columns[:col_num] + ['...'] + columns[-col_num:]

        if num_rows > options.max_rows:
            first: List[int] = list(range(options.max_rows // 2))
            last: List[int] = list(range(num_rows - options.max_rows // 2,
                                         num_rows))
            idx: List[int] = first + last
        else:
            idx = list(range(num_rows))

        # Gets integer row number data
        data_list: List[List[str]] = [[''] + [str(i) for i in idx]]
        long_len: List[int] = [len(data_list[0][-1])]
        decimal_len: List[int] = [0]
        data: List
        cur_len: int
        dec_len: List
        whole_len: List
        dec_len_arr: ndarray
        whole_len_arr: ndarray

        for column in columns:
            if column != '...':
                vals = self._get_column_values(column)[idx]
                dtype = self._column_info[column].dtype
                if dtype == 'M':
                    unit = utils.get_datetime_str(vals)
                    vals = vals.astype(f'datetime64[{unit}]')
                    data = [column] + [str(val).replace('T', ' ') if not np.isnat(val)
                                       else str(val) for val in vals]
                elif dtype == 'm':
                    unit = utils.get_timedelta_str(vals)
                    vals = vals.astype(f'timedelta64[{unit}]')
                    data = [column] + [str(val).replace('T', ' ') if not np.isnat(val)
                                       else str(val) for val in vals]
                elif dtype == 'S':
                    loc = self._column_info[column].loc
                    rev_map = self._str_reverse_map[loc]
                    data = [column] + ['NaN' if val == 0 else rev_map[val] for val in vals]
                elif dtype == 'i':
                    data = [column] + ['NaN' if val == MIN_INT else val for val in vals]
                elif dtype == 'b':
                    bool_dict = {-1: 'NaN', 0: 'False', 1: 'True'}
                    data = [column] + [bool_dict[val] for val in vals]
                elif dtype == 'f':
                    data = [column] + ['NaN' if np.isnan(val) else val for val in vals]
            else:
                data = ['...'] * (len(idx) + 1)
                long_len.append(3)
                decimal_len.append(0)
                data_list.append(data)
                continue

            if len(self) == 0:
                data_list.append(data)
                long_len.append(len(column))
                decimal_len.append(0)
                continue

            if self._column_info[column].dtype == 'S':
                cur_len = max([len(str(x)) for x in data])
                cur_len = min(cur_len, options.max_colwidth)
                long_len.append(cur_len)
                decimal_len.append(0)
            elif self._column_info[column].dtype == 'f':
                dec_len = [utils.get_decimal_len(x) for x in data[1:]]
                whole_len = [utils.get_whole_len(x) for x in data[1:]]

                dec_len_arr = np.array(dec_len).clip(0, 6)
                whole_len_arr = np.array(whole_len)
                lengths = [len(column), dec_len_arr.max() + whole_len_arr.max() + 1]

                max_decimal = dec_len_arr.max()
                long_len.append(max(lengths))
                decimal_len.append(min(max_decimal, 6))
            elif self._column_info[column].dtype == 'i':
                lengths = [len(column)] + [len(str(x)) for x in data[1:]]
                long_len.append(max(lengths))
                decimal_len.append(0)
            elif self._column_info[column].dtype == 'b':
                long_len.append(max(len(column), 5))
                decimal_len.append(0)
            elif self._column_info[column].dtype in 'Mm':
                long_len.append(max(len(column), len(data[1])))
                decimal_len.append(0)

            data_list.append(data)

        return data_list, long_len, decimal_len, idx

    def __repr__(self) -> str:
        data_list: List[List[str]]
        decimal_len: List[int]
        idx: List[int]
        data_list, long_len, decimal_len, idx = self._build_repr()
        return_string: str = ''

        for i in range(len(data_list[0])):
            for j in range(len(data_list)):
                d = data_list[j][i]
                fl = long_len[j]
                dl = decimal_len[j]

                if isinstance(d, str):
                    cur_str = d
                    if len(cur_str) > options.max_colwidth:
                        cur_str = cur_str[:options.max_colwidth - 3] + "..."
                    return_string += f'{cur_str: >{fl}}  '
                else:
                    return_string += f'{d: >{fl}.{dl}f}  '

            if i == options.max_rows // 2 and len(self) > options.max_rows:
                return_string += '\n'
                for j, fl in enumerate(long_len):
                    return_string += f'{"...": >{str(fl)}}'
            return_string += '\n'
        return return_string

    def _repr_html_(self) -> str:
        data_list: List[List[str]]
        decimal_len: List[int]
        idx: List[int]
        data_list, long_len, decimal_len, idx = self._build_repr()

        return_string: str = '<table>'
        for i in range(len(idx) + 1):
            if i == 0:
                return_string += '<thead>'
            elif i == 1:
                return_string += '<tbody>'
            return_string += '<tr>'
            for j, (d, fl, dl) in enumerate(zip(data_list, long_len, decimal_len)):
                if str(d[i]) == 'nan':
                    d[i] = 'NaN'
                if d[i] is None:
                    d[i] = 'None'

                ts = '<th>' if j * i == 0 else '<td>'
                te = '</th>' if j * i == 0 else '</td>'
                if isinstance(d[i], bool):
                    d[i] = str(d[i])
                if isinstance(d[i], str):
                    cur_str = d[i]
                    if len(cur_str) > options.max_colwidth:
                        cur_str = cur_str[:options.max_colwidth - 3] + "..."
                    return_string += f'{ts}{cur_str: >{fl}}{te}'
                else:
                    return_string += f'{ts}{d[i]: >{fl}.{dl}f}{te}'
            if i == options.max_rows // 2 and len(self) > options.max_rows:
                return_string += '<tr>'
                for j, fl in enumerate(long_len):
                    ts = '<th>' if j == 0 else '<td>'
                    te = '</th>' if j == 0 else '</td>'
                    return_string += f'{ts}{"...": >{str(fl)}}{te}'
                return_string += '</tr>'
            return_string += '</tr>'
            if i == 0:
                return_string += '</thead>'
        return return_string + '</tbody></table>'

    def __len__(self) -> int:
        for _, arr in self._data.items():
            return len(arr)
        return 0

    @property
    def shape(self) -> Tuple[int, int]:
        """
        Returns the number of rows and columns as a two-item tuple
        """
        return len(self), len(self._columns)

    @property
    def size(self) -> int:
        return len(self) * len(self._columns)

    def _get_col_name_from_int(self, iloc: int) -> str:
        try:
            return self._columns[iloc]
        except IndexError:
            raise IndexError(f'Index {iloc} is out of bounds for the columns')

    def _get_list_of_cols_from_selection(self, cs: Iterable[Any]) -> List[str]:
        new_cols: List[str] = []
        bool_count: int = 0

        for i, col in enumerate(cs):
            if isinstance(col, bool):
                bool_count += 1
                if col:
                    new_cols.append(self._get_col_name_from_int(i))
                continue

            if isinstance(col, (int, np.integer)):
                new_cols.append(self._get_col_name_from_int(col))
            elif col not in self._column_info:
                raise KeyError(f'{col} is not in the columns')
            else:
                new_cols.append(col)
            if bool_count > 0:
                raise TypeError('Your column selection has booleans mixed with other types. '
                                'You can only use booleans by themselves.')

        if bool_count > 0:
            if bool_count != self.shape[1]:
                raise ValueError('The length of the boolean list must match the number of '
                                 f'columns {i} != {self.shape[1]}')

        utils.check_duplicate_list(new_cols)
        return new_cols

    def _find_col_location(self, col: str) -> int:
        try:
            return self._column_info[col].order
        except KeyError:
            raise KeyError(f'{col} is not in the columns')

    def _convert_col_sel(self, cs: ColSel) -> List[str]:
        if isinstance(cs, str):
            self._validate_column_name(cs)
            return [cs]
        if isinstance(cs, int):
            return [self._get_col_name_from_int(cs)]
        if isinstance(cs, slice):
            sss: ListIntNone = []
            for s in ['start', 'stop', 'step']:
                value: Union[str, Optional[int]] = getattr(cs, s)
                if value is None or isinstance(value, int):
                    sss.append(value)
                elif isinstance(value, str):
                    if s == 'step':
                        raise TypeError('Slice step must be None or int')
                    sss.append(self._find_col_location(value))
                else:
                    raise TypeError('Slice start, stop, and step values must '
                                    'be int, str, or None')
            if isinstance(cs.stop, str):
                if cs.step is None or cs.step > 0:
                    sss[1] += 1
                elif cs.step < 0:
                    sss[1] -= 1
            return self._columns[slice(*sss)]
        if isinstance(cs, list):
            return self._get_list_of_cols_from_selection(cs)
        if isinstance(cs, ndarray):
            col_array: ndarray = utils.try_to_squeeze_array(cs)
            if col_array.dtype.kind == 'b':
                if len(col_array) != self.shape[1]:
                    raise ValueError('Length of column selection boolean '
                                     'array must be the same as number of '
                                     'columns in the DataFrame. '
                                     f'{len(col_array)} != {self.shape[1]}')
                return self._columns[col_array]
            else:
                return self._get_list_of_cols_from_selection(col_array)
        elif isinstance(cs, DataFrame):
            if cs.shape[0] != 1:
                raise ValueError('Boolean selection only works with single-'
                                 'row DataFames')
            col_array = cs.values.squeeze()
            if col_array.dtype.kind != 'b':
                raise TypeError('All values for column selection must '
                                'be boolean')
            if len(col_array) != self.shape[1]:
                raise ValueError('Number of booleans in DataFrame does not '
                                 'equal number of columns in self '
                                 f'{len(col_array)} != {self.shape[1]}')
            return self._columns[col_array]
        else:
            raise TypeError('Selection must either be one of '
                            'int, str, list, array, slice or DataFrame')

    def _convert_row_sel(self, rs: RowSel) -> Union[List[int], ndarray]:
        if isinstance(rs, slice):
            def check_none_int(obj: Any) -> bool:
                return obj is None or isinstance(obj, int)

            all_ok: bool = (check_none_int(rs.start) and
                            check_none_int(rs.stop) and
                            check_none_int(rs.step))

            if not all_ok:
                raise TypeError('Slice start, stop, and step values must be int or None')
        elif isinstance(rs, list):
            # check length of boolean list is length of rows
            bool_count: int = 0
            for i, row in enumerate(rs):
                if isinstance(row, bool):
                    bool_count += 1
                    continue
                # self.columns is a list to prevent numpy warning
                if not isinstance(row, int):
                    raise TypeError('Row selection must consist only of integers')

                if bool_count > 0:
                    raise TypeError('Your row selection has a mix of boolean and integers. They '
                                    'must be either all booleans or all integers.')

            if bool_count > 0 and bool_count != len(self):
                raise ValueError('Length of boolean array must be the same as DataFrame. '
                                 f'{len(rs)} != {len(self)}')
        elif isinstance(rs, ndarray):
            row_array: ndarray = utils.try_to_squeeze_array(rs)
            if row_array.dtype.kind == 'b':
                if len(row_array) != len(self):
                    raise ValueError('Length of boolean array must be the same as DataFrame. '
                                     f'{len(rs)} != {len(self)}')
            elif row_array.dtype.kind != 'i':
                raise TypeError('Row selection array data type must be either integer or boolean')
            rs = row_array
        elif isinstance(rs, DataFrame):
            if rs.shape[0] != 1 and rs.shape[1] != 1:
                raise ValueError('When using a DataFrame for selecting rows, it must have '
                                 'either one row or one column')
            if rs.shape[0] == 1:
                row_array = rs.values[0, :]
            else:
                row_array = rs.values[:, 0]
            if row_array.dtype.kind not in 'bi':
                if row_array.dtype.kind == 'f':
                    # check if float can be safely converted to int
                    if (row_array % 1).sum() == 0:
                        row_array = row_array.astype('int64')
                    else:
                        raise ValueError('Your DataFrame for row selection has float values that '
                                         'are not whole numbers. Use a DataFrame with only '
                                         'integers or booleans.')
                else:
                    raise TypeError('All values for row selection must be boolean or integer')
            rs = row_array
        elif isinstance(rs, int):
            rs = [rs]
        else:
            raise TypeError('Selection must either be one of '
                            'int, list, array, slice, or DataFrame')
        return rs

    def _getitem_scalar(self, rs: int, cs: Union[int, str]) -> Scalar:
        # most common case, string column, integer row
        try:
            dtype, loc = self._get_col_dtype_loc(cs)  # type: str, int
        # if its not a key error then it will be an index error
        # for the rows and raise the default exception message
        except KeyError:
            # if its a key error, then column not found
            # must check if string or integer
            if isinstance(cs, str):
                raise KeyError(f'{cs} is not a column')
            else:
                # if column is integer, could still be valid
                try:
                    cs = self._columns[cs]
                except IndexError:
                    raise IndexError(f'Column integer location {cs} is '
                                     'out of range from the columns')
                # now we know column is valid
                dtype, loc = self._get_col_dtype_loc(cs)  # type: str, int

        if dtype == 'S':
            val = self._data[dtype][rs, loc]
            return self._str_reverse_map[loc][val]
        else:
            return self._data[dtype][rs, loc]

    def _select_entire_single_column(self, col_sel):
        self._validate_column_name(col_sel)
        dtype, loc = self._get_col_dtype_loc(col_sel)  # type: str, int
        new_data = {dtype: self._data[dtype][:, loc].reshape(-1, 1)}
        new_columns = np.array([col_sel], dtype='O')
        new_column_info = {col_sel: utils.Column(dtype, 0, 0)}
        new_str_reverse_map = {}
        if dtype == 'S':
            new_str_reverse_map = {0: self._str_reverse_map[loc]}
        return self._construct_from_new(new_data, new_column_info, new_columns, new_str_reverse_map)

    def _construct_df_from_selection(self, rs: Union[List[int], ndarray, slice],
                                     cs: List[str]) -> 'DataFrame':
        new_data: Dict[str, ndarray] = {}
        dt_positions: Dict[str, List[int]] = defaultdict(list)
        new_column_info: ColInfoT = {}
        new_columns = np.asarray(cs, dtype='O')
        new_str_reverse_map: StrRevMap = {}

        for i, col in enumerate(cs):
            dtype, loc = self._get_col_dtype_loc(col)  # type: str, int
            cur_loc: int = len(dt_positions[dtype])
            new_column_info[col] = utils.Column(dtype, cur_loc, i)
            dt_positions[dtype].append(loc)
            if dtype == 'S':
                new_str_reverse_map[cur_loc] = self._str_reverse_map[loc]

        for dtype, pos in dt_positions.items():
            if isinstance(rs, (list, ndarray)):
                ix = np.ix_(rs, pos)
                arr = self._data[dtype][ix]
            else:
                # otherwise it is a slice
                arr = self._data[dtype][rs, pos]

            if arr.ndim == 1:
                arr = arr[:, np.newaxis]

            if dtype == 'S':
                new_str_reverse_map, arr = _va.bool_selection_str_mapping(arr, new_str_reverse_map)

            new_data[dtype] = np.asfortranarray(arr)

        return self._construct_from_new(new_data, new_column_info, new_columns, new_str_reverse_map)

    @overload
    def __getitem__(self, value: Tuple[List, slice]) -> 'DataFrame':
        pass

    @overload
    def __getitem__(self, value: Tuple[slice, List]) -> 'DataFrame':
        pass

    @overload
    def __getitem__(self, value: Tuple['DataFrame', slice]) -> 'DataFrame':
        pass

    @overload
    def __getitem__(self, value: Tuple[slice, 'DataFrame']) -> 'DataFrame':
        pass

    def __getitem__(self, value: Tuple[RowSel,  ColSel]) -> Union[Scalar, 'DataFrame']:
        """
        Selects a subset of the DataFrame. Must always pass a row and column selection.

        Row selection must be either a list or array of integers, a one column boolean DataFrame,
            or a slice
        Column Selection can be a mix of column names and integers, a boolean
        Parameters
        ----------
        value : A tuple consisting of both a row and a column selection

        Returns
        -------
        A new DataFrame

        """
        utils.validate_selection_size(value)
        row_sel, col_sel = value  # type: RowSel, ColSel
        if utils.is_integer(row_sel) and isinstance(col_sel, (int, str)):
            return self._getitem_scalar(row_sel, col_sel)

        elif utils.is_entire_column_selection(row_sel, col_sel):
            return self._select_entire_single_column(col_sel)
        else:
            row_sel = self._convert_row_sel(row_sel)
            col_sel = self._convert_col_sel(col_sel)
            return self._construct_df_from_selection(row_sel, col_sel)

    def to_dict(self, orient: str = 'array') -> Dict[str, Union[ndarray, List]]:
        """
        Convert DataFrame to dictionary of 1-dimensional arrays or lists

        Parameters
        ----------
        orient : str {'array' or 'list'}
        Determines the type of the values of the dictionary.
        """
        if orient not in ['array', 'list']:
            raise ValueError('orient must be either "array" or "list"')
        data: Dict[str, Union[ndarray, List]] = {}

        col: str
        for col in self._columns:
            arr = self._get_column_values(col)
            if orient == 'array':
                data[col] = arr.copy()
            else:
                data[col] = arr.tolist()
        return data

    def _is_numeric_or_bool(self) -> bool:
        return set(self._data.keys()) <= set('bif')

    def _is_string(self) -> bool:
        return set(self._data.keys()) == {'S'}

    def _is_date(self) -> bool:
        return set(self._data.keys()) <= {'m', 'M'}

    def _has_numeric_or_bool(self) -> bool:
        """
        Does this DataFrame have at least one numeric columns?
        """
        dtypes: Set[str] = set(self._data.keys())
        return 'i' in dtypes or 'f' in dtypes or 'b' in dtypes

    def _has_numeric_strict(self) -> bool:
        """
        Does this DataFrame have at least one numeric columns?
        """
        return bool({'i', 'f'} & self._data.keys())

    def _has_string(self) -> bool:
        return 'S' in self._data

    def _copy_column_info(self) -> ColInfoT:
        new_column_info = {}
        for col, dtype, loc, order in self._col_info_iter(with_order=True):
            new_column_info[col] = utils.Column(dtype, loc, order)
        return new_column_info

    def _is_empty(self):
        """
        Check if DataFrame has 0 total elements.
        Used before calling many other methods

        Returns
        -------
        bool
        """
        return self.size == 0

    def copy(self) -> 'DataFrame':
        """
        Returns an exact replica of the DataFrame as a copy

        Returns
        -------
        A copy of the DataFrame

        """
        new_data: Dict[str, ndarray] = {dt: arr.copy() for dt, arr in self._data.items()}
        new_columns: ColumnT = self._columns.copy()
        new_column_info: ColInfoT = self._copy_column_info()
        new_str_reverse_map = deepcopy(self._str_reverse_map)
        return self._construct_from_new(new_data, new_column_info, new_columns, new_str_reverse_map)

    def select_dtypes(self, include: Optional[Union[str, List[str]]] = None,
                      exclude: Optional[Union[str, List[str]]] = None, copy=False) -> 'DataFrame':
        """
        Selects columns based on their data type.
        The data types must be passed as strings - 'int', 'float', 'str', 'bool'.
        You may pass either a single string or a list of strings to either `include` or `exclude`
        Use the string 'number' to select both ints and floats simultaneously

        Parameters
        ----------
        include - str or list of strings to include in returned DataFrame
        exclude - str or list of strings to include in returned DataFrame

        Returns
        -------
        A DataFrame of just the included or excluded types.

        Notes
        -----
        If both include and exclude are not none an exception is raised

        Examples
        --------
        >>> df.select_dtypes('int')
        >>> df.select_dtypes('number')
        >>> df.select_dtypes(['float', 'str'])
        >>> df.select_dtypes(exclude=['bool', 'str'])
        """
        if include is not None and exclude is not None:
            raise ValueError('Provide only one of either include/exclude')
        if include is None and exclude is None:
            return self.copy()

        clude: Set[str] = utils.convert_clude(include or exclude)
        include_final: Set[str]
        if include:
            include_final = self._data.keys() & clude
        else:
            include_final = self._data.keys() - clude

        if copy:
            new_data: Dict[str, ndarray] = {dtype: self._data[dtype].copy('F')
                                            for dtype in include_final}
        else:
            new_data: Dict[str, ndarray] = {dtype: self._data[dtype] for dtype in include_final}

        new_column_info: ColInfoT = {}
        new_columns: ColumnT = []
        new_str_reverse_map: StrRevMap = {}
        order: int = 0
        for col, dtype, loc in self._col_info_iter():  # type: str, str, int
            if dtype in include_final:
                new_column_info[col] = utils.Column(dtype, loc, order)
                new_columns.append(col)
                order += 1
        if 'S' in include_final:
            new_str_reverse_map = self._str_reverse_map
            if copy:
                new_str_reverse_map = deepcopy(new_str_reverse_map)
        new_columns = np.asarray(new_columns, dtype='O')
        return self._construct_from_new(new_data, new_column_info, new_columns, new_str_reverse_map)

    @classmethod
    def _construct_from_new(cls: Type[object], data: Dict[str, ndarray],
                            column_info: ColInfoT, columns: ColumnT,
                            str_reverse_map: StrRevMap) -> 'DataFrame':
        df_new: 'DataFrame' = object.__new__(cls)
        df_new._column_info = column_info
        df_new._data = data
        df_new._columns = columns
        df_new._hasnans = {}
        df_new._str_reverse_map = str_reverse_map
        return df_new

    def _op_scalar_eval(self, other: Any, op_string: str) -> Tuple[DictListArr, ColInfoT]:
        data_dict: DictListArr = defaultdict(list)
        kind_shape: Dict[str, Tuple[str, int]] = {}
        arr_res: ndarray
        new_str_reverse_map: StrRevMap = {}

        for old_kind, arr in self._data.items():
            if old_kind == 'S':
                func_name = f'str{op_string}'
                if hasattr(_mos, func_name):
                    func = getattr(_mos, func_name)
                    new_str_reverse_map, arr_res, new_kind = func(self._str_reverse_map, self._data['S'], other)
                else:
                    raise TypeError('Operation does not work on string columns')

            else:
                with np.errstate(invalid='ignore', divide='ignore'):
                    # TODO: do something about zero division error
                    arr_res = getattr(arr, op_string)(other)
                new_kind = arr_res.dtype.kind

            cur_len: int = utils.get_num_cols(data_dict.get(new_kind, []))
            kind_shape[old_kind] = (new_kind, cur_len)
            data_dict[new_kind].append(arr_res)

        new_column_info: ColInfoT = {}
        for col, col_obj in self._column_info.items():
            old_kind, old_loc, order = col_obj.values  # type: str, int, int
            new_kind, new_loc = kind_shape[old_kind]  # type: str, int
            new_column_info[col] = utils.Column(new_kind, new_loc + old_loc, order)

        new_data: Dict[str, ndarray] = utils.concat_stat_arrays(data_dict)
        return self._construct_from_new(new_data, new_column_info, self._columns.copy(),
                                        new_str_reverse_map)

    def _op_df(self, other: Any, op_string: str) -> 'DataFrame':
        op = OP_2D(self, other, op_string)
        return op.operate()

    def _op(self, other: Any, op_string: str) -> 'DataFrame':
        if utils.is_scalar(other):
            return self._op_scalar_eval(other, op_string)
        elif isinstance(other, DataFrame):
            return self._op_df(other, op_string)
        elif isinstance(other, ndarray):
            return self._op_df(DataFrame(other), op_string)
        else:
            raise TypeError('other must be int, float, str, bool, timedelta, '
                            'datetime, array or DataFrame')

    def __add__(self, other: Any) -> 'DataFrame':
        return self._op(other, '__add__')

    def __radd__(self, other: Any) -> 'DataFrame':
        return self._op(other, '__radd__')

    def __mul__(self, other: Any) -> 'DataFrame':
        return self._op(other, '__mul__')

    def __rmul__(self, other: Any) -> 'DataFrame':
        return self._op(other, '__rmul__')

    def __sub__(self, other: Any) -> 'DataFrame':
        return self._op(other, '__sub__')

    def __rsub__(self, other: Any) -> 'DataFrame':
        return self._op(other, '__rsub__')

    def __truediv__(self, other: Any) -> 'DataFrame':
        return self._op(other, '__truediv__')

    def __rtruediv__(self, other: Any) -> 'DataFrame':
        return self._op(other, '__rtruediv__')

    def __floordiv__(self, other: Any) -> 'DataFrame':
        return self._op(other, '__floordiv__')

    def __rfloordiv__(self, other: Any) -> 'DataFrame':
        return self._op(other, '__rfloordiv__')

    def __pow__(self, other: Any) -> 'DataFrame':
        return self._op(other, '__pow__')

    def __rpow__(self, other: Any) -> 'DataFrame':
        return self._op(other, '__rpow__')

    def __mod__(self, other: Any) -> 'DataFrame':
        return self._op(other, '__mod__')

    def __rmod__(self, other: Any) -> 'DataFrame':
        return self._op(other, '__rmod__')

    def __gt__(self, other: Any) -> 'DataFrame':
        return self._op(other, '__gt__')

    def __ge__(self, other: Any) -> 'DataFrame':
        return self._op(other, '__ge__')

    def __lt__(self, other: Any) -> 'DataFrame':
        return self._op(other, '__lt__')

    def __le__(self, other: Any) -> 'DataFrame':
        return self._op(other, '__le__')

    def __eq__(self, other: Any) -> 'DataFrame':
        return self._op(other, '__eq__')

    def __ne__(self, other: Any) -> 'DataFrame':
        return self._op(other, '__ne__')

    def __neg__(self) -> 'DataFrame':
        if self._is_numeric_or_bool():
            new_data: Dict[str, ndarray] = {}
            for dt, arr in self._data.items():
                new_data[dt] = -arr
        else:
            raise TypeError('Only works for all numeric columns')
        new_column_info: ColInfoT = self._copy_column_info()
        return self._construct_from_new(new_data, new_column_info, self._columns.copy(), {}, {})

    def __bool__(self) -> NoReturn:
        raise ValueError(': The truth value of an array with more than one element is ambiguous. '
                         'Use a.any() or a.all()')

    def __getattr__(self, name: str) -> NoReturn:
        raise AttributeError("'DataFrame' object has no "
                             f"attribute '{name}'")

    def __iadd__(self, value: Any) -> NoReturn:
        raise NotImplementedError(f'Use df = df + {value}')

    def __isub__(self, value: Any) -> NoReturn:
        raise NotImplementedError(f'Use df = df - {value}')

    def __imul__(self, value: Any) -> NoReturn:
        raise NotImplementedError(f'Use df = df * {value}')

    def __itruediv__(self, value: Any) -> NoReturn:
        raise NotImplementedError(f'Use df = df / {value}')

    def __ifloordiv__(self, value: Any) -> NoReturn:
        raise NotImplementedError(f'Use df = df // {value}')

    def __imod__(self, value: Any) -> NoReturn:
        raise NotImplementedError(f'Use df = df % {value}')

    def __ipow__(self, value: Any) -> NoReturn:
        raise NotImplementedError(f'Use df = df ** {value}')

    @property
    def dtypes(self) -> 'DataFrame':
        """
        Returns a two column name DataFrame with the name of each column and
        its correspond data type
        """
        dtype_list: List[str] = [utils.convert_kind_to_dtype(self._column_info[col].dtype)
                                 for col in self._columns]
        arr: ndarray = np.array(dtype_list, dtype='O')
        columns: List[str] = ['Column Name', 'Data Type']
        data, str_reverse_map = _va.convert_str_to_cat_list_2d([self._columns, arr])
        new_data: Dict[str, ndarray] = {'S': data}
        new_column_info: ColInfoT = {'Column Name': utils.Column('S', 0, 0),
                                     'Data Type': utils.Column('S', 1, 1)}
        return self._construct_from_new(new_data, new_column_info, np.array(columns, dtype='O'),
                                        str_reverse_map)

    def __and__(self, other: Any) -> 'DataFrame':
        return self._op_logical(other, '__and__')

    def __or__(self, other: Any) -> 'DataFrame':
        return self._op_logical(other, '__or__')

    def _validate_matching_shape(self, other: Union[ndarray, 'DataFrame']) -> None:
        if isinstance(other, DataFrame):
            oshape: Tuple[int, int] = other.shape
        elif isinstance(other, ndarray):
            if other.ndim == 1:
                oshape = (len(other), 1)
            else:
                oshape = other.shape

        if self.shape != oshape:
            raise ValueError('Shape of left DataFrame does not match shape of right '
                             f'{self.shape} != {oshape}')

    def _op_logical(self, other: Union[ndarray, 'DataFrame'], op_logical: str) -> 'DataFrame':
        if not isinstance(other, (DataFrame, ndarray)):
            d = {'__and__': '&', '__or__': '|'}
            raise TypeError(f'Must use {d[op_logical]} operator with either DataFrames or arrays.')

        self._validate_matching_shape(other)
        new_data: Dict[str, ndarray] = {}

        arr_left: ndarray = self.values

        if isinstance(other, DataFrame):
            arr_right: ndarray = other.values
        else:
            arr_right = other

        if arr_left.dtype.kind != 'b' or arr_left.dtype.kind != 'b':
            raise TypeError(f'The logical operator {d[op_logical]} only works with boolean arrays'
                            'or DataFrames')

        new_data['b'] = getattr(arr_left, op_logical)(arr_right)
        new_column_info: ColInfoT = {col: utils.Column('b', i, i)
                                     for i, col in enumerate(self._columns)}

        return self._construct_from_new(new_data, new_column_info, self._columns.copy(), {})

    def __invert__(self) -> 'DataFrame':
        if set(self._data) == {'b'}:
            new_data: Dict[str, ndarray] = {dt: ~arr for dt, arr in self._data.items()}
            new_column_info: ColInfoT = self._copy_column_info()
            return self._construct_from_new(new_data, new_column_info, self._columns.copy(), {})
        else:
            raise TypeError('Invert operator only works on DataFrames with all boolean columns')

    def _astype_internal(self, column: str, numpy_dtype: str) -> None:
        """
        Changes one column dtype in-place
        """
        new_kind: str = utils.convert_numpy_to_kind(numpy_dtype)
        dtype, loc, order = self._get_col_dtype_loc_order(column)  # type: str, int, int

        srm = []

        if dtype == new_kind:
            return None
        col_data: ndarray = self._data[dtype][:, loc]
        nulls = utils.isna_array(col_data, dtype)

        if numpy_dtype == 'S':
            col_data = col_data.astype('U')
            col_data, _, srm = _va.convert_str_to_cat(col_data)
            col_data[nulls] = 0
        elif numpy_dtype == 'b':
            col_data = col_data.astype('bool').astype('int8')
            col_data[nulls] = -1
        elif numpy_dtype == 'i':
            col_data = col_data.astype('int64')
            col_data[nulls] = MIN_INT
        elif numpy_dtype == 'f':
            col_data = col_data.astype('int64')
            col_data[nulls] = np.nan
        elif col_data.dtype.kind == 'M':
            col_data = col_data.astype('datetime64[ns]')
            col_data[nulls] = NaT
        elif col_data.dtype.kind == 'm':
            col_data = col_data.astype('timedelta64[ns]')
            col_data[nulls] = NaT

        self._remove_column(column)
        self._write_new_column_data(column, new_kind, col_data, srm, order)

    def _remove_column(self, column: str) -> None:
        """
        Removes column from _colum_dtype, and _data
        Keeps column name in _columns
        """
        dtype, loc, order = self._column_info.pop(column).values
        self._data[dtype] = np.delete(self._data[dtype], loc, axis=1)
        if self._data[dtype].shape[1] == 0:
            del self._data[dtype]

        for col, col_obj in self._column_info.items():
            if col_obj.dtype == dtype and col_obj.loc > loc:
                col_obj.loc -= 1

    def _write_new_column_data(self, column: str, new_kind: str, data: ndarray,
                               srm: list, order: int) -> None:
        """
        Adds data to _data, the data type info but does not
        append name to columns
        """
        if new_kind not in self._data:
            loc = 0
        else:
            loc = self._data[new_kind].shape[1]
        if new_kind == 'S':
            self._str_reverse_map[loc] = srm
        self._column_info[column] = utils.Column(new_kind, loc, order)
        if new_kind in self._data:
            self._data[new_kind] = np.asfortranarray(np.column_stack((self._data[new_kind], data)))
        else:
            if data.ndim == 1:
                data = data[:, np.newaxis]

            self._data[new_kind] = np.asfortranarray(data)

    def _add_new_column(self, column: str, kind: str, data: ndarray, srm) -> None:
        order: int = len(self._columns)
        self._write_new_column_data(column, kind, data, srm, order)
        self._columns = np.append(self._columns, column)

    def _full_columm_add(self, column: str, kind: str, data: ndarray, srm: list) -> None:
        """
        Either adds a brand new column or
        overwrites an old column
        """
        if column not in self._column_info:
            self._add_new_column(column, kind, data, srm)
        # column is in df
        else:
            # data type has changed
            dtype, loc, order = self._get_col_dtype_loc_order(column)  # type: str, int, int
            if dtype != kind:
                self._remove_column(column)
                self._write_new_column_data(column, kind, data, srm, order)
            # data type same as original
            else:
                self._data[kind][:, loc] = data
                if kind == 'S':
                    self._str_reverse_map[loc] = srm

    def __setitem__(self, key: Any, value: Any) -> None:
        utils.validate_selection_size(key)

        # row selection and column selection
        rs, cs = key  # type: RowSel, ColSel
        if isinstance(rs, int) and isinstance(cs, (int, str)):
            return self._setitem_single_scalar(rs, cs, value)

        # select an entire column or new column
        if utils.is_entire_column_selection(rs, cs):
            # ignore type check here. Function ensures cs is a string
            return self._setitem_entire_column(cs, value)  # type: ignore

        col_list: List[str] = self._convert_col_sel(cs)
        row_list: Union[List[int], ndarray] = self._convert_row_sel(rs)

        if utils.is_scalar(value):
            self._setitem_multiple_scalar(row_list, col_list, value)
        else:
            self._setitem_multiple_multiple(row_list, col_list, value)

    def _setitem_single_scalar(self, rs: RowSel, cs: Union[int, str], value: Scalar) -> None:
        """
        Assigns a scalar to exactly a single cell
        """
        if isinstance(value, (DataFrame, ndarray)):
            if value.size != 1:
                raise ValueError('When setting with a DataFrame or array, '
                                 'there must be only one value in it')
            if isinstance(value, DataFrame):
                value = value[0, 0]
            elif value.ndim == 1:
                value = value[0]
            elif value.ndim == 2:
                value = value[0, 0]
            else:
                raise ValueError('numpy array must be either 1 or 2 dimensions')

        if isinstance(cs, str):
            self._validate_column_name(cs)
            col_name: str = cs
        else:
            col_name = self._get_col_name_from_int(cs)

        dtype, loc = self._get_col_dtype_loc(col_name)  # type: str, int
        value = self._setitem_change_dtype(value, dtype, loc, col_name)
        dtype, loc = self._get_col_dtype_loc(col_name)  # type: str, int
        if dtype == 'S':
            # update
            old_codes = self._data[dtype][:, loc]
            old_srm = self._str_reverse_map[loc]
            old_code = old_codes[rs]
            old_code_exists = old_code in old_codes[:rs] or old_code in old_codes[rs + 1:]
            new_value = False
            try:
                value = old_srm.index(value)
            except ValueError:
                new_value = True
            if old_code_exists:
                if new_value:
                    old_srm.append(value)
                    value = len(old_srm) - 1
                old_codes[rs] = value
            else:
                # old string doesn't exist any more
                if new_value:
                    old_srm[old_code] = value
                else:
                    old_srm.pop(old_code)
                    old_codes[old_codes > old_code] -= 1
                    if value > old_code:
                        value -= 1
                    old_codes[rs] = value
        else:
            self._data[dtype][rs, loc] = value

    def _setitem_change_dtype(self, value, dtype: str, loc: int, col_name: str):
        is_bool = isinstance(value, (bool, np.bool_))
        is_int = isinstance(value, (int, np.integer))
        is_float = isinstance(value, (float, np.floating))
        is_dt = isinstance(value, np.datetime64)
        is_td = isinstance(value, np.timedelta64)
        is_str = isinstance(value, (bytes, str))
        is_nan = is_float and np.isnan(value)
        is_none = value is None
        bad_type = False

        if dtype == 'b':
            if is_bool:
                value = int(value)
            elif is_nan or is_none:
                value = -1
            elif is_int:
                self._astype_internal(col_name, 'int64')
            elif is_float and not np.isnan(value):
                self._astype_internal(col_name, 'float64')
            else:
                bad_type = True
        elif dtype == 'i':
            if is_int:
                pass
            elif is_nan or is_none:
                value = MIN_INT
            elif is_float:
                self._astype_internal(col_name, 'float64')
            else:
                bad_type = True
        elif dtype == 'f':
            if not (is_float or is_int or is_bool or is_none):
                bad_type = True
        elif dtype == 'M':
            if not is_dt:
                bad_type = True
        elif dtype == 'm':
            if not is_td:
                bad_type = True
        elif dtype == 'S':
            if is_nan or is_none:
                value = False
            elif not is_str:
                bad_type = True

        if bad_type:
            raise TypeError(f'Cannot set column {col_name} which has type {utils._DT[dtype]} '
                            f'with type {type(value)}')
        return value

    def _setitem_entire_column(self, cs: str, value: Union[Scalar, ndarray, 'DataFrame']) -> None:
        # TODO: Change to string
        """
        Called when setting an entire column (old or new)
        df[:, 'col'] = value
        """
        srm = []
        if utils.is_scalar(value):
            arr: ndarray = np.repeat(value, len(self))
            kind = arr.dtype.kind
        elif isinstance(value, list):
            utils.validate_array_size(value, len(self))
            arr = value
            kind = 'O'
        elif isinstance(value, ndarray):
            utils.validate_array_size(value, len(self))
            arr = utils.try_to_squeeze_array(value)
            kind = arr.dtype.kind
        elif isinstance(value, DataFrame):
            if value.shape[0] != self.shape[0]:
                raise ValueError(f'The DataFrame on the left has {self.shape[0]} rows. '
                                 f'The DataFrame on the right has {self.shape[0]} rows. '
                                 'They must be equal')
            if value.shape[1] != 1:
                raise ValueError('You are setting exactly one column. The DataFrame you are '
                                 f'trying to set this with has {value.shape[1]} columns. '
                                 'They must be equal')
            col = value.columns[0]
            kind, loc, _ = value._column_info[col].values
            arr = value._data[kind][:, loc]
            if kind == 'S':
                srm = value._str_reverse_map[loc]
            self._full_columm_add(cs, kind, arr, srm)
        else:
            raise TypeError('Must use a scalar, a list, an array, or a '
                            'DataFrame when setting new values')

        if kind == 'O':
            arr, kind, srm = _va.convert_object_array(arr, cs)
        elif kind == 'b':
            arr = arr.astype('int8')
        elif kind in 'SU':
            arr = arr.astype('U')
            arr, kind, srm = _va.convert_str_to_cat(arr)
        elif kind == 'M':
            arr = arr.astype('datetime64[ns]')
        elif kind == 'm':
            arr = arr.astype('timedelta64[ns]')
        self._full_columm_add(cs, kind, arr, srm)

    def _setitem_multiple_scalar(self, rows: Union[List[int], ndarray], cols: List[str], value: Any) -> None:
        """
        Sets new data when the selection is a list or array. Could be boolean array.
        Not assigning a single column
        """
        cur_kinds = [self._column_info[col].dtype for col in cols]
        value_kind = _va.get_kind(value)
        if value_kind == 'unknown':
            raise TypeError(f'Cannot set new value {value} which has type {type(value)}')
        utils.setitem_validate_scalar_col_types(cur_kinds, value_kind, cols)

        for col, dtype, loc, col_arr in self._col_info_iter(False, True):  # type: str, str, int
            if dtype == 'i' and value_kind == 'f':
                self._astype_internal(col, 'float64')
                dtype, loc = self._get_col_dtype_loc(col)  # type: str, int

            if value_kind == 'missing':
                value = utils.get_missing_value_code(dtype)
            elif value_kind == 'b':
                value = int(value)

            if value_kind == 'S':
                srm = self._str_reverse_map[loc]
                codes = col_arr
                str_map = {False: 0}
                new_srm = [False]
                new_codes = np.empty(len(codes), 'uint32', 'F')
                if isinstance(rows, ndarray) and rows.dtype.kind == 'b':
                    for i, code in enumerate(codes):
                        if rows[i]:
                            cur_str = value
                        else:
                            cur_str = srm[code]
                        n_before = len(str_map)
                        new_codes[i] = str_map.setdefault(cur_str, len(str_map))
                        n_after = len(str_map)
                        if n_after > n_before:
                            new_srm.append(cur_str)
                else:
                    j = 0
                    for i, code in enumerate(codes):
                        if j < len(rows) and rows[j] == i:
                            cur_str = value
                            j += 1
                        else:
                            cur_str = srm[code]
                        n_before = len(str_map)
                        new_codes[i] = str_map.setdefault(cur_str, len(str_map))
                        n_after = len(str_map)
                        if n_after > n_before:
                            new_srm.append(cur_str)

                self._str_reverse_map[loc] = new_srm
                self._data[dtype][:, loc] = new_codes

            else:
                self._data[dtype][rows, loc] = value

    def _setitem_multiple_multiple(self, rows: Union[List[int], ndarray], cols: List[str],
                                         value: Any) -> None:
        cur_kinds = [self._column_info[col].dtype for col in cols]
        nrows_to_set, ncols_to_set = self._setitem_nrows_ncols_to_set(rows, cols)  # type: int, int
        arrs: List[ndarray] = []
        kinds = []
        srms = []
        if isinstance(value, list):
            value = utils.convert_lists_vertical(value)
            # need to update convert_to_arrays to cover missing values
            arrs, kinds, srms = utils.convert_to_arrays(value, ncols_to_set, cur_kinds)
        elif isinstance(value, ndarray):
            arrs, kinds, srms = utils.convert_to_arrays(value, ncols_to_set, cur_kinds)
        elif isinstance(value, DataFrame):
            for col in value._columns:
                srm = []
                kind, loc, _ = value._column_info[col].values
                arrs.append(value._data[kind][:, loc])
                kinds.append(kind)
                if kind == 'S':
                    srm = value._str_reverse_map[loc]
                srms.append(srm)
        else:
            raise TypeError('Must use a scalar, a list, an array, or a '
                            'DataFrame when setting new values')
        utils.setitem_validate_shape(nrows_to_set, ncols_to_set, arrs)
        utils.setitem_validate_col_types(cur_kinds, kinds, cols)
        self._setitem_non_scalar(rows, cols, arrs, cur_kinds, kinds, srms)

    def _setitem_nrows_ncols_to_set(self, rs: Union[List[int], ndarray],
                                    cs: List[str]) -> Tuple[int, int]:
        if isinstance(rs, int):
            nrows: int = 1
        else:
            nrows = len(np.arange(len(self))[rs])
        ncols: int = len(cs)
        return nrows, ncols

    def _setitem_non_scalar(self, rows: Union[List[int], ndarray], cols: List[str],
                            arrs: List[ndarray], kinds1: List[str], kinds2: List[str],
                            srms: List[List]) -> None:
        for col, arr, k1, k2, srm in zip(cols, arrs, kinds1, kinds2, srms):  # type: str, ndarray, str, str, List
            if k1 == 'i' and k2 == 'f':
                dtype_internal: str = utils.convert_kind_to_numpy(k2)
                self._astype_internal(col, dtype_internal)
            dtype, loc = self._get_col_dtype_loc(col)  # type: str, int
            if dtype == 'S':
                cur_srm = self._str_reverse_map[loc]
                for val in srm:
                    if val not in cur_srm:
                        cur_srm.append(val)
                arr = [cur_srm.index(srm[code]) for code in arr]

            else:
                self._data[dtype][rows, loc] = arr

    def _validate_column_name(self, column: str) -> None:
        if column not in self._column_info:
            raise KeyError(f'Column "{column}" does not exist')

    def _validate_column_name_list(self, columns: list) -> None:
        col_set: Set[str] = set()
        for col in columns:
            self._validate_column_name(col)
            if col in col_set:
                raise ValueError(f'"{col}" has already been selected as a column')
            col_set.add(col)

    def _new_col_info_from_kind_shape(self, kind_shape: Dict[str, int], new_kind: str) -> ColInfoT:
        new_column_info: ColInfoT = {}
        for col, dtype, loc, order in self._col_info_iter(with_order=True):  # type: str, str, int, int
            add_loc: int = kind_shape[dtype]
            new_column_info[col] = utils.Column(new_kind, loc + add_loc, order)
        return new_column_info

    def astype(self, dtype: Union[Dict[str, str], str]) -> 'DataFrame':
        """
        Changes the data type of one or more columns. Valid data types are
        int, float, bool, str. Change all the columns as once by passing a string, otherwise
        use a dictionary of column names mapped to their data type

        Parameters
        ----------
        dtype : str or list of strings

        Returns
        -------
        New DataFrame with new data types

        """
        if isinstance(dtype, str):
            new_dtype: str = utils.check_valid_dtype_convert(dtype)
            data_dict: DictListArr = defaultdict(list)
            kind_shape: Dict[str, int] = OrderedDict()
            total_shape: int = 0

            old_kind: str
            arr: ndarray
            for old_kind, arr in self._data.items():
                kind_shape[old_kind] = total_shape
                total_shape += arr.shape[1]

                if new_dtype != old_kind:
                    nanable = False
                    if old_kind == 'f':
                        nanable = True
                        na_arr = np.isnan(arr)
                    elif old_kind == 'S':
                        nanable = True
                        na_arr = _math.isna_str(arr, np.zeros(len(arr), dtype='bool'))
                    elif old_kind in 'mM':
                        nanable = True
                        na_arr = np.isnat(arr)

                    if new_dtype == 'S':
                        arr = arr.astype('U').astype('O')
                        if nanable:
                            arr[na_arr] = None
                    elif new_dtype == 'M':
                        arr = arr.astype(new_dtype).astype('datetime64[ns]')
                    elif new_dtype == 'm':
                        arr = arr.astype(new_dtype).astype('timedelta64[ns]')
                    elif new_dtype == 'f':
                        arr = arr.astype(new_dtype)
                        if nanable:
                            arr[na_arr] = nan
                    else:
                        arr = arr.astype(new_dtype)

                new_kind = arr.dtype.kind
                data_dict[new_kind].append(arr)

            new_column_info: ColInfoT = self._new_col_info_from_kind_shape(kind_shape, new_kind)
            new_data: Dict[str, ndarray] = utils.concat_stat_arrays(data_dict)
            return self._construct_from_new(new_data, new_column_info, self._columns.copy())

        elif isinstance(dtype, dict):
            df_new: 'DataFrame' = self.copy()

            for column, new_dtype2 in dtype.items():  # type: str, str
                df_new._validate_column_name(column)
                new_dtype_numpy: str = utils.check_valid_dtype_convert(new_dtype2)
                df_new._astype_internal(column, new_dtype_numpy)
            return df_new
        else:
            raise TypeError('Argument dtype must be either a string or a dictionary')

    def head(self, n: int = 5) -> 'DataFrame':
        """
        Select the first `n` rows

        Parameters
        ----------
        n : int

        """
        return self[:n, :]  # type: ignore

    def tail(self, n: int = 5) -> 'DataFrame':
        """
        Selects the last n rows

        Parameters
        ----------
        n: int
        """
        return self[-n:, :]  # type: ignore

    @property
    def hasnans(self) -> 'DataFrame':
        """
        Returns a new two-column DataFrame with the column names in one column
        and a boolean value in the other alerting whether any missing values exist
        """
        if self._hasnans == {}:
            for kind, arr in self._data.items():  # type: str, ndarray
                if kind in 'f':
                    self._hasnans['f'] = np.isnan(arr).any(0)
                elif kind == 'S':
                    self._hasnans['S'] = _va.isnan_object_2d(arr)
                elif kind in 'mM':
                    self._hasnans[kind] = np.isnat(arr).any(0)
                else:
                    self._hasnans[kind] = np.zeros(arr.shape[1], dtype='bool')

        bool_array: ndarray = np.empty(len(self), dtype='bool')
        for col, dtype2, loc, order in self._col_info_iter(with_order=True):  # type: str, str, int, int
            bool_array[order] = self._hasnans[dtype2][loc]

        columns: ndarray = np.array(['Column Name', 'Has NaN'])
        new_data: Dict[str, ndarray] = {'S': self._columns[:, np.newaxis],
                                        'b': bool_array[:, np.newaxis]}

        new_column_info: ColInfoT = {'Column Name': utils.Column('S', 0, 0),
                                     'Has NaN': utils.Column('b', 0, 1)}
        return self._construct_from_new(new_data, new_column_info, columns)

    def _get_stat_dtypes_axis0(self, name: str, include_strings: bool = True) -> Set[str]:
        dtypes = set(self._data)
        if not include_strings:
            dtypes -= {'S'}

        if name in ['mean', 'median', 'quantile']:
            dtypes -= {'M', 'S'}
        elif name in ['sum', 'cumsum']:
            dtypes -= {'M'}
        elif name in ['std', 'var', 'prod', 'cumprod']:
            dtypes -= {'m', 'M', 'S'}

        if not dtypes:
            raise ValueError('There are no columns in this DataFrame that '
                             f'operate with the `{name}` method')
        return dtypes

    def _check_stat_dtypes_axis1(self, name: str) -> None:
        for dtype in self._data:
            if name not in stat.funcs[dtype]:
                raise TypeError(f'Cannot compute {name} method on DataFrame with data type '
                                f'{utils._DT[dtype]} when axis="columns"')
            if len(self._data) > 1 and dtype in 'OMm' and name not in {'count', 'nunique'}:
                dt_name = utils._DT[dtype]
                raise TypeError(f'The `{name}` method cannot work with axis="columns" '
                                'when the the DataFrame contains a mix of '
                                f'{dt_name} and non-{dt_name} values')

    def _get_stat_func_result(self, kind: str, name: str, arr: ndarray,
                              axis: int, kwargs: Dict) -> ndarray:
        func_name: str = utils.get_stat_func_name(name, kind)
        func: Callable = getattr(_math, func_name)
        hasnans = self._hasnans_dtype(kind)
        kwargs.update({'hasnans': hasnans})
        if kind == 'S':
            # when is this returning a scalar?
            result: Union[Scalar, ndarray] = func(arr, self._str_reverse_map, axis=axis, **kwargs)
        else:
            result = func(arr, axis=axis, **kwargs)

        if isinstance(result, ndarray):
            arr = result
        else:
            arr = np.array([result])

        if arr.ndim == 1:
            if axis == 0:
                arr = arr[np.newaxis, :]
            else:
                arr = arr[:, np.newaxis]
        return arr

    def _stat_funcs_axis0(self, name: str, include_strings: bool, **kwargs: Any) -> 'DataFrame':
        data_dict: DictListArr = defaultdict(list)
        new_column_info: ColInfoT = {}
        change_kind: Dict[str, Tuple[str, int]] = {}
        new_num_cols: int = 0
        dtypes: Set[str] = self._get_stat_dtypes_axis0(name, include_strings)
        keep = name in ['cumsum', 'cummin', 'cummax', 'cumprod']
        new_str_reverse_map = {}

        for kind, arr in self._data.items():  # type: str, ndarray
            if kind in dtypes:
                func_name: str = utils.get_stat_func_name(name, kind)
                func: Callable = getattr(_math, func_name)
                hasnans = self._hasnans_dtype(kind)
                kwargs.update({'hasnans': hasnans})
                if kind == 'S':
                    return_obj = func(arr, self._str_reverse_map, axis=0, **kwargs)
                else:
                    return_obj = func(arr, axis=0, **kwargs)
                if isinstance(return_obj, tuple):
                    arr, new_str_reverse_map = return_obj
                    new_kind: str = 'S'
                else:
                    arr = return_obj
                    new_kind = arr.dtype.kind
            elif not keep:
                continue
            else:
                new_kind = arr.dtype.kind

            if arr.ndim == 1:
                arr = arr[np.newaxis, :]
            cur_loc: int = utils.get_num_cols(data_dict.get(new_kind, []))
            change_kind[kind] = (new_kind, cur_loc)
            data_dict[new_kind].append(arr)
            new_num_cols += arr.shape[1]

        new_columns: ndarray = np.empty(new_num_cols, dtype='O')
        i: int = 0

        for col in self._columns:  # type: str
            kind, loc, order = self._column_info[col].values  # type: str, int, int
            if kind in dtypes or keep:
                new_columns[i] = col
                new_kind, add_loc = change_kind[kind]  # type: str, int
                new_column_info[col] = utils.Column(new_kind, loc + add_loc, i)
                i += 1

        new_data: Dict[str, ndarray] = utils.concat_stat_arrays(data_dict)
        return self._construct_from_new(new_data, new_column_info, new_columns, new_str_reverse_map)

    def _stat_funcs_axis1(self, name, **kwargs: Any) -> 'DataFrame':
        self._check_stat_dtypes_axis1(name)
        arrs: List[ndarray] = []
        new_column_info: ColInfoT = {}

        if utils.is_column_stack_func(name):
            # TODO: Optimize this and divide function into those where order matters and
            # where it does not mean vs cumsum. Right now, self.values preserves order
            arr = self.values
            kind = arr.dtype.kind
            result: ndarray = self._get_stat_func_result(kind, name, arr, 1, kwargs)
        else:
            for kind, arr in self._data.items():
                arrs.append(self._get_stat_func_result(kind, name, arr, 1, kwargs))

            if len(arrs) == 1:
                result = arrs[0]
            else:
                func_across: Callable = stat.funcs_columns[name]
                result = func_across(arrs)

        if utils.is_agg_func(name):
            new_columns = np.array([name], dtype='O')
        else:
            new_columns = self._columns.copy()

        new_kind = result.dtype.kind
        new_data = {new_kind: result}

        for i, col in enumerate(new_columns):
            new_column_info[col] = utils.Column(new_kind, i, i)
        return self._construct_from_new(new_data, new_column_info, new_columns)

    def _stat_funcs(self, name: str, axis: str, include_strings: bool = True,
                    **kwargs: Any) -> 'DataFrame':
        axis_int: int = utils.convert_axis_string(axis)
        if axis_int == 0:
            return self._stat_funcs_axis0(name, include_strings, **kwargs)
        else:
            return self._stat_funcs_axis1(name, **kwargs)

    def sum(self, axis: str = 'rows', include_strings: bool = False) -> 'DataFrame':
        return self._stat_funcs('sum', axis, include_strings)

    def prod(self, axis: str = 'rows') -> 'DataFrame':
        return self._stat_funcs('prod', axis)

    def max(self, axis: str = 'rows') -> 'DataFrame':
        return self._stat_funcs('max', axis)

    def min(self, axis: str = 'rows') -> 'DataFrame':
        return self._stat_funcs('min', axis)

    def mean(self, axis: str = 'rows') -> 'DataFrame':
        return self._stat_funcs('mean', axis)

    def median(self, axis: str = 'rows') -> 'DataFrame':
        return self._stat_funcs('median', axis)

    def std(self, axis: str = 'rows', ddof: int = 1) -> 'DataFrame':
        return self._stat_funcs('std', axis, ddof=ddof)

    def var(self, axis: str = 'rows', ddof: int = 1) -> 'DataFrame':
        return self._stat_funcs('var', axis, ddof=ddof)

    def count(self, axis: str = 'rows') -> 'DataFrame':
        return self._stat_funcs('count', axis)

    def cummax(self, axis: str = 'rows') -> 'DataFrame':
        return self._stat_funcs('cummax', axis)

    def cummin(self, axis: str = 'rows') -> 'DataFrame':
        return self._stat_funcs('cummin', axis)

    def cumsum(self, axis: str = 'rows') -> 'DataFrame':
        return self._stat_funcs('cumsum', axis)

    def cumprod(self, axis: str = 'rows') -> 'DataFrame':
        return self._stat_funcs('cumprod', axis)

    def any(self, axis: str = 'rows') -> 'DataFrame':  # todo add for ft
        return self._stat_funcs('any', axis)

    def all(self, axis: str = 'rows') -> 'DataFrame':
        return self._stat_funcs('all', axis)

    def argmax(self, axis: str = 'rows') -> 'DataFrame':
        """
        Returns the integer location of the maximum value of each column
        When setting `axis` to 'columns', all the non-numeric/bool columns are dropped first

        Parameters
        ----------
        axis : str - either 'rows' or 'columns'
        """
        return self._stat_funcs('argmax', axis)

    def argmin(self, axis: str = 'rows') -> 'DataFrame':
        """
        Returns the integer location of the minimum value of each column
        When setting `axis` to 'columns', all the non-numeric/bool columns are dropped first

        Parameters
        ----------
        axis : str - either 'rows' or 'columns'
        """
        return self._stat_funcs('argmin', axis)

    def mode(self, axis: str = 'rows', keep='low') -> 'DataFrame':
        """
        Returns the mode of each column or row

        Parameters
        ----------
        axis: 'rows' or 'columns'

        keep: 'low' or 'high'
            When there are ties, chose either the lower or higher of the
            values.

        Returns
        -------
        A DataFrame
        """
        if keep not in ('low', 'high', 'all'):
            raise ValueError('`keep` must be either "low", "high", or "all"')
        return self._stat_funcs('mode', axis, keep=keep)

    def _non_agg_stat_funcs(self, name: str, **kwargs: Any) -> 'DataFrame':
        new_data: Dict[str, ndarray] = {}
        for dt, arr in self._data.items():  # type: str, ndarray
            # abs value possible for timedelta but not round
            if dt in 'if' or (dt == 'm' and name == 'abs'):
                new_data[dt] = getattr(np, name)(arr, **kwargs)
            else:
                new_data[dt] = arr.copy()
        new_column_info: ColInfoT = self._copy_column_info()
        return self._construct_from_new(new_data, new_column_info, self._columns.copy())

    def abs(self) -> 'DataFrame':
        return self._non_agg_stat_funcs('abs')

    def round(self, decimals: int = 0) -> 'DataFrame':
        if not utils.is_integer(decimals):
            raise TypeError('`decimals` must be an integer')
        return self._non_agg_stat_funcs('round', decimals=decimals)

    __abs__ = abs
    __round__ = round

    def _get_clip_dtype(self, value: Any, name: str) -> str:
        if value is None:
            raise ValueError('You must provide a value for either lower or upper')
        if utils.is_number(value):
            if self._has_numeric_or_bool():
                return 'number'
            else:
                raise TypeError(f'You provided a numeric value for {name} '
                                'but do not have any numeric columns')
        elif isinstance(value, str):
            if self._has_string():
                return 'str'
            else:
                raise TypeError(f'You provided a string value for {name} '
                                'but do not have any string columns')
        else:
            raise NotImplementedError('Data type incompatible')

    def clip(self, lower: Optional[ScalarT] = None, upper: Optional[ScalarT] = None) -> 'DataFrame':
        if lower is None:
            overall_dtype = self._get_clip_dtype(upper, 'upper')  # type: str
        elif upper is None:
            overall_dtype = self._get_clip_dtype(lower, 'lower')
        else:
            overall_dtype = utils.is_compatible_values(lower, upper)
            if lower > upper:
                raise ValueError('The upper value must be less than lower')

        new_data: Dict[str, ndarray] = {}
        df = self
        if overall_dtype == 'str':
            if lower is None:
                new_data['S'] = _math.clip_str_upper(self._data['S'], upper)
            elif upper is None:
                new_data['S'] = _math.clip_str_lower(self._data['S'], lower)
            else:
                new_data['S'] = _math.clip_str_both(self._data['S'], lower, upper)

            for kind, arr in self._data.items():
                if kind != 'S':
                    new_data[kind] = arr
        else:
            if utils.is_float(lower) or utils.is_float(upper):
                as_type_dict = {col: 'float' for col, col_obj in self._column_info.items()
                                if col_obj.dtype == 'i'}
                df = df.astype(as_type_dict)
            for dtype, arr in df._data.items():
                if dtype in 'if':
                    new_data[dtype] = arr.clip(lower, upper)
                else:
                    new_data[dtype] = arr.copy(order='F')

        new_column_info: ColInfoT = df._copy_column_info()
        return df._construct_from_new(new_data, new_column_info, df._columns.copy())

    def _hasnans_dtype(self, kind: str) -> ndarray:
        try:
            return self._hasnans[kind]
        except KeyError:
            return np.ones(self._data[kind].shape[1], dtype='bool')

    def isna(self) -> 'DataFrame':
        """
        Returns a DataFrame of booleans the same size as the original indicating whether or not
        each value is NaN or not.
        """
        new_data = {'b': np.empty(self.shape, 'int8', 'F')}
        new_column_info: ColInfoT = {}

        for i, (col, _, _, col_arr) in enumerate(self._col_info_iter(with_arr=True)):  # type: str, str, int, int
            new_data[:, i] = utils.isna_array(col_arr)
            new_column_info[col] = utils.Column('b', i, i)

        return self._construct_from_new(new_data, new_column_info, self._columns.copy(), {})

    def quantile(self, axis: str = 'rows', q: float = 0.5) -> 'DataFrame':
        """
        Computes the quantile of each numeric/boolean column

        Parameters
        ----------
        axis : 'rows' or 'columns'
        q : a float between 0 and 1

        Returns
        -------
        A DataFrame of the quantile for each column/row
        """
        if not utils.is_number(q):
            raise TypeError('`q` must be a number between 0 and 1')
        if q < 0 or q > 1:
            raise ValueError('`q` must be between 0 and 1')
        return self._stat_funcs('quantile', axis, q=q)

    def _get_dtype_list(self) -> ndarray:
        dtypes = [utils.convert_kind_to_dtype(self._column_info[col].dtype)
                  for col in self._columns]
        return np.array(dtypes, dtype='O')

    def _null_pct(self) -> 'DataFrame':
        return self.isna().mean()

    def describe(self, percentiles: List[float] = [.25, .5, .75],
                 summary_type: str = 'numeric') -> 'DataFrame':
        """
        Provides several summary statistics for each column
        Parameters
        ----------
        percentiles
        summary_type

        Returns
        -------

        """
        if summary_type == 'numeric':
            df = self.select_dtypes('number')
        elif summary_type == 'non-numeric':
            df = self.select_dtypes(['str', 'bool', 'datetime'])
        else:
            raise ValueError('`summary_type` must be either "numeric" or "non-numeric"')
        data_dict: DictListArr = defaultdict(list)
        new_column_info: ColInfoT = {}
        new_columns: List[str] = []

        if summary_type == 'numeric':

            data_dict['S'].append(df._columns.copy('F'))
            new_column_info['Column Name'] = utils.Column('S', 0, 0)
            new_columns.append('Column Name')

            dtypes = df._get_dtype_list()
            data_dict['S'].append(dtypes)
            new_column_info['Data Type'] = utils.Column('S', 1, 1)
            new_columns.append('Data Type')

            funcs: List[Tuple[str, Tuple[str, Dict]]] = [('count', ('i', {})),
                                                         ('_null_pct', ('f', {})),
                                                         ('mean', ('f', {})),
                                                         ('std', ('f', {})),
                                                         ('min', ('f', {}))]

            for perc in percentiles:
                funcs.append(('quantile', ('f', {'q': perc})))

            funcs.append(('max', ('f', {})))

            change_name: Dict[str, Callable] = {'_null_pct': lambda x: 'null %',
                                                'quantile': lambda x: f"{x['q'] * 100:.2g}%"}

            order: int = 1
            for func_name, (dtype, kwargs) in funcs:
                loc: int = len(data_dict[dtype])
                order += 1
                value: ndarray = getattr(df, func_name)(**kwargs).values[0]
                data_dict[dtype].append(value)
                name = change_name.get(func_name, lambda x: func_name)(kwargs)
                new_column_info[name] = utils.Column(dtype, loc, order)
                new_columns.append(name)

            new_data: Dict[str, ndarray] = init.concat_arrays(data_dict)

        else:
            raise NotImplementedError('non-numeric summary not available yet')

        return self._construct_from_new(new_data, new_column_info,
                                        np.asarray(new_columns, dtype='O'))

    def dropna(self, axis: str = 'rows', how: str = 'any', thresh: Union[int, float] = None,
               subset: List[IntStr] = None) -> 'DataFrame':
        """
        Drops rows/columns if there is 1 or more missing value for that row/column
        Change `how` to 'all' to only drop rows/columns when all values are missing

        Use `thresh` to specify the percentage or count of non-missing values needed to keep
            the row/colum

        Parameters
        ----------
        axis : 'rows' or 'columns' - rows will drop rows, columns will drop columns
        how : 'any' or 'all'
        thresh : A float between 0 and 1 or an integer
        subset : A list of columns or rows to limit your search for NaNs. Only this subset will be
            considered.

        """
        axis = utils.swap_axis_name(axis)
        df: 'DataFrame'
        if subset is not None:
            if axis == 'rows':
                df = self[subset, :]
            elif axis == 'columns':
                df = self[:, subset]
        else:
            df = self

        if thresh is not None:
            if isinstance(thresh, int):
                criteria = (~df.isna()).sum(axis) >= thresh
            elif isinstance(thresh, float):
                if thresh < 0 or thresh > 1:
                    raise ValueError('thresh must either be an integer or a float between 0 and 1')
                criteria = (~df.isna()).mean(axis) >= thresh
            else:
                raise TypeError('thresh must be an integer or a float between 0 and 1')
        elif how == 'any':
            criteria = ~df.isna().any(axis)
        elif how == 'all':
            criteria = ~df.isna().all(axis)
        else:
            raise ValueError('how must be either "any" or "all"')

        if axis == 'rows':
            return self[:, criteria]
        return self[criteria, :]

    # noinspection PyPep8Naming
    def cov(self) -> 'DataFrame':
        """
        Computes the covariance between each column.

        Returns
        -------
        An n x n DataFrame where n is the number of columns
        """
        if self._is_string():
            raise TypeError('DataFrame consists only of strings. Must have int, float, '
                            'or bool columns')

        x: ndarray = self._values_number()
        if x.dtype.kind == 'i':
            x0: ndarray = x[0]
            x_diff: ndarray = x - x0
            Exy: ndarray = (x_diff.T @ x_diff)
            Ex: ndarray = x_diff.sum(0)[np.newaxis, :]
            ExEy: ndarray = Ex.T @ Ex
            counts: Union[int, ndarray] = len(x)
        else:
            x0 = _math.get_first_non_nan(x)
            x_diff = x - x0
            x_not_nan: ndarray = (~np.isnan(x)).astype(int)

            x_diff_0: ndarray = np.nan_to_num(x_diff)
            counts = (x_not_nan.T @ x_not_nan)
            Exy = (x_diff_0.T @ x_diff_0)
            Ex = (x_diff_0.T @ x_not_nan)
            ExEy = Ex * Ex.T

        with np.errstate(invalid='ignore'):
            cov: ndarray = (Exy - ExEy / counts) / (counts - 1)

        new_data: Dict[str, ndarray] = {'f': np.asfortranarray(cov)}
        new_column_info: ColInfoT = {'Column Name': utils.Column('S', 0, 0)}
        new_columns: ndarray = np.empty(x.shape[1] + 1, dtype='O')
        new_columns[0] = 'Column Name'

        i: int = 0
        for col, dtype, loc in self._col_info_iter():  # type: str, str, int
            if dtype not in 'ifb':
                continue
            new_column_info[col] = utils.Column('f', i, i + 1)
            new_columns[i + 1] = col
            i += 1
        new_data['S'] = np.asfortranarray(new_columns[1:])[:, np.newaxis]
        return self._construct_from_new(new_data, new_column_info,
                                        np.asarray(new_columns, dtype='O'))

    def corr(self) -> 'DataFrame':
        """
        Computes the correlation between each column. Only does pearson correlation for now.

        Returns
        -------
        An n x n DataFrame where n is the number of columns
        """
        if self._is_string():
            raise TypeError('DataFrame consists only of strings. Must have int, float, '
                            'or bool columns')

        x: ndarray = self._values_number()
        if x.dtype.kind == 'i':
            x0: ndarray = x[0]
            x_diff: ndarray = x - x0
            Exy: ndarray = (x_diff.T @ x_diff)
            Ex: ndarray = x_diff.sum(0)[np.newaxis, :]
            ExEy: ndarray = Ex.T @ Ex
            counts: Union[int, ndarray] = len(x)
            Ex2: ndarray = (x_diff ** 2).sum(0)

        else:
            x0 = _math.get_first_non_nan(x)
            x_diff = x - x0
            x_not_nan: ndarray = (~np.isnan(x)).astype(int)

            # get index of first non nan too and check for nan here
            x_diff_0: ndarray = np.nan_to_num(x_diff)
            counts = (x_not_nan.T @ x_not_nan)
            Exy = (x_diff_0.T @ x_diff_0)
            Ex = (x_diff_0.T @ x_not_nan)
            ExEy = Ex * Ex.T
            Ex2 = (x_diff_0.T ** 2 @ x_not_nan)

        with np.errstate(invalid='ignore'):
            cov: ndarray = (Exy - ExEy / counts) / (counts - 1)
            stdx: ndarray = (Ex2 - Ex ** 2 / counts) / (counts - 1)
            stdxy: ndarray = stdx * stdx.T
            corr: ndarray = cov / np.sqrt(stdxy)

        new_data: Dict[str, ndarray] = {'f': np.asfortranarray(corr)}
        new_column_info: ColInfoT = {'Column Name': utils.Column('S', 0, 0)}
        new_columns: ndarray = np.empty(x.shape[1] + 1, dtype='O')
        new_columns[0] = 'Column Name'

        i: int = 0
        for col, dtype, loc in self._col_info_iter():  # type: str, str, int
            if dtype not in 'ifb':
                continue
            new_column_info[col] = utils.Column('f', i, i + 1)
            new_columns[i + 1] = col
            i += 1
        new_data['S'] = np.asfortranarray(new_columns[1:])[:, np.newaxis]
        return self._construct_from_new(new_data, new_column_info,
                                        np.asarray(new_columns, dtype='O'))

    def unique(self, subset: Union[str, List[str], None] = None, only_subset: bool = False,
               keep: str = 'first') -> ndarray:
        """
        Finds the unique elements of a single column in the order that they appeared

        Parameters
        ----------
        subset :
        only_subset :
        keep :

        Returns
        -------
        A one dimensional NumPy array
        """
        if subset is None:
            subset = self.columns
        elif isinstance(subset, str):
            self._validate_column_name(subset)
            subset = [subset]
        elif isinstance(subset, list):
            self._validate_column_name_list(subset)
        else:
            raise TypeError('`subset` must be None, a column name as a string, '
                            'or column names in a list')

        if not isinstance(only_subset, (bool, np.bool_)):
            raise TypeError('`only_subset` must be a boolean')

        if keep not in ('first', 'last', 'none'):
            raise ValueError('`keep` must be either "first", "last", or "none"')

        def keep_all(arr_keep: ndarray) -> 'DataFrame':
            new_data: Dict[str, ndarray] = {}
            for dtype, arr in self._data.items():  # type: str, ndarray
                new_data[dtype] = arr[np.ix_(arr_keep)]
            new_columns: ndarray = self._columns.copy()
            new_column_info: ColInfoT = self._copy_column_info()
            return self._construct_from_new(new_data, new_column_info, new_columns)

        def keep_subset(arr_keep: ndarray,
                        dtype_col: Dict[str, List[str]],
                        dtype_loc: Dict[str, List[int]],
                        new_columns: ndarray,
                        new_col_order: Dict[str, int]) -> 'DataFrame':

            new_data: Dict[str, ndarray] = {}
            new_column_info: ColInfoT = {}
            for dtype, locs in dtype_loc.items():  # type: str, List[int]
                arr: ndarray = self._data[dtype]
                if arr.shape[1] == len(locs):
                    new_data[dtype] = arr[arr_keep]
                    cur_locs: List[int] = locs
                else:
                    arr_new: ndarray = arr[np.ix_(arr_keep, locs)]
                    if arr_new.ndim == 1:
                        arr_new = arr_new[:, np.newaxis]
                    new_data[dtype] = arr_new
                    cur_locs = list(range(len(locs)))

                for col, loc in zip(dtype_col[dtype], cur_locs):
                    new_column_info[col] = utils.Column(dtype, loc, new_col_order[col])
            return self._construct_from_new(new_data, new_column_info, new_columns)

        if keep == 'none':
            # todo: have special cases for single columns/dtypes
            dtype_col, dtype_loc, new_columns, new_col_order = self._get_all_dtype_info_subset(
                subset)

            arrs: List[ndarray] = []
            has_obj: bool = False
            has_nums: bool = False
            for dtype, locs in dtype_loc.items():
                arr: ndarray = self._data[dtype]
                if keep == 'last':
                    arr = arr[::-1]
                if dtype == 'S':
                    has_obj = True
                    if len(locs) != arr.shape[1]:
                        arr_obj: ndarray = arr[:, locs]
                    else:
                        arr_obj = arr
                else:
                    has_nums = True
                    if len(locs) != arr.shape[1]:
                        arrs.append(arr[:, locs])
                    else:
                        arrs.append(arr)

            if has_nums:
                if len(arrs) > 1:
                    arr_numbers: ndarray = np.column_stack(arrs)
                else:
                    arr_numbers = arrs[0]
                arr_numbers = np.ascontiguousarray(arr_numbers)
                if arr_numbers.ndim == 2 and arr_numbers.shape[1] == 1:
                    arr_numbers = arr_numbers.squeeze(1)
                dtype = arr_numbers.dtype.kind

            if has_obj:
                if arr_obj.ndim == 2 and arr_obj.shape[1] == 1:
                    arr_obj = arr_obj.squeeze(1)

            if has_obj and not has_nums:
                if arr_obj.ndim == 1:
                    arr_keep = _uq.unique_str_none(arr_obj)
                else:
                    arr_keep = _uq.unique_str_none_2d(arr_obj)
            elif not has_obj and has_nums:
                func_name: str = 'unique_' + utils.convert_kind_to_dtype(dtype) + '_none'
                if arr_numbers.ndim == 2:
                    func_name += '_2d'
                arr_keep = getattr(_uq, func_name)(arr_numbers)
            else:
                func_name = 'unique_' + utils.convert_kind_to_dtype(dtype) + '_string_none'
                if arr_numbers.ndim == 2:
                    func_name += '_2d'
                if arr_obj.ndim == 1:
                    arr_obj = arr_obj[:, np.newaxis]
                arr_keep = getattr(_uq, func_name)(arr_numbers, arr_obj)

            if only_subset:
                return keep_subset(arr_keep, dtype_col, dtype_loc, new_columns, new_col_order)
            else:
                return keep_all(arr_keep)

        if len(subset) == 1:
            col: str = subset[0]
            kind, loc, _ = self._column_info[col].values  # type: str, int, int
            arr = self._data[kind][:, loc]

            if keep == 'last':
                arr = arr[::-1]

            if kind in 'mM':
                nans = np.isnat(arr)
                arr_keep = _uq.unique_date(arr.view('int64'), nans)
            else:
                func_name = 'unique_' + utils.convert_kind_to_dtype(kind)
                arr_keep = getattr(_uq, func_name)(arr)

            if keep == 'last':
                arr_keep = arr_keep[::-1]

            if only_subset:
                new_data = {kind: arr[np.ix_(arr_keep)][:, np.newaxis]}
                new_columns = np.array([col], dtype='object')
                new_column_info = {col: utils.Column(kind, 0, 0)}
                return self._construct_from_new(new_data, new_column_info, new_columns)
            else:
                return keep_all(arr_keep)
        else:
            # returns columns in order from df regardless of subset order
            dtype_col, dtype_loc, new_columns, new_col_order = self._get_all_dtype_info_subset(
                subset)
            dtypes = list(dtype_col.keys())

            if len(dtypes) == 1:
                dtype = dtypes[0]
                locs = dtype_loc[dtype]
                arr = self._data[dtype]

                if arr.shape[1] != len(locs):
                    arr = arr[:, locs]
                    locs = list(range(len(locs)))

                arr = np.ascontiguousarray(arr)

                if keep == 'last':
                    arr = arr[::-1]

                if dtype in 'mM':
                    arr_keep = _uq.unique_date_2d(arr.view('int64'))
                else:
                    func_name = 'unique_' + utils.convert_kind_to_dtype(dtype) + '_2d'
                    arr_keep = getattr(_uq, func_name)(arr)

                if keep == 'last':
                    arr_keep = arr_keep[::-1]

                new_column_info = {}
                if only_subset:
                    # no need to check if 1 dim to add np.newaxis, since there are always
                    # a min of 2 columns here
                    new_data = {dtype: arr[np.ix_(arr_keep)]}
                    new_columns = dtype_col[dtype]
                    new_columns = np.array(new_columns, dtype='object')
                    for col, loc in zip(new_columns, locs):
                        new_column_info[col] = utils.Column(dtype, loc, new_col_order[col])
                    return self._construct_from_new(new_data, new_column_info, new_columns)
                else:
                    return keep_all(arr_keep)
            else:
                arrs = []
                has_obj = False
                for dtype, locs in dtype_loc.items():
                    arr = self._data[dtype]
                    if dtype in 'mM':
                        arr = arr.view('int64')

                    if keep == 'last':
                        arr = arr[::-1]
                    if dtype == 'S':
                        has_obj = True
                        if len(locs) != arr.shape[1]:
                            arr_obj = arr[:, locs]
                        else:
                            arr_obj = arr
                    else:
                        if len(locs) != arr.shape[1]:
                            arrs.append(arr[:, locs])
                        else:
                            arrs.append(arr)

                if len(arrs) > 1:
                    arr_numbers = np.column_stack(arrs)
                else:
                    arr_numbers = arrs[0]

                arr_numbers = np.ascontiguousarray(arr_numbers)
                dtype = arr_numbers.dtype.kind

                if not has_obj:
                    func_name = 'unique_' + utils.convert_kind_to_dtype(dtype)
                    if arr_numbers.ndim == 2:
                        func_name += '_2d'
                    arr_keep = getattr(_uq, func_name)(arr_numbers)
                else:
                    func_name = 'unique_' + utils.convert_kind_to_dtype(dtype) + '_string'
                    if arr_numbers.shape[1] == 1:
                        arr_numbers = arr_numbers.squeeze(1)
                    else:
                        func_name += '_2d'
                    arr_keep = getattr(_uq, func_name)(arr_numbers, arr_obj)

                if keep == 'last':
                    arr_keep = arr_keep[::-1]

                if only_subset:
                    return keep_subset(arr_keep, dtype_col, dtype_loc, new_columns, new_col_order)
                else:
                    return keep_all(arr_keep)

    def nunique(self, axis: str = 'rows', count_na: bool = False) -> 'DataFrame':
        """
        Counts the number of unique elements for each column/row

        Parameters
        ----------
        axis : 'rows' or 'columns'
        count_na : bool - When True, NaN will be counted at most once for each row/column.
                            When False, NaNs will be ignored.

        Returns
        -------
        A one column/row DataFrame
        """
        return self._stat_funcs('nunique', axis, count_na=count_na)

    def fillna(self,
               values: Union[Scalar, Dict[str, Scalar], None] = None,
               method: Optional[str] = None,
               limit: Optional[int] = None,
               fill_function: Optional[str] = None) -> 'DataFrame':
        """

        Parameters
        ----------
        values: number, string or  dictionary of column name to fill value
        method : {'bfill', 'ffill'}
        limit : positive integer
        fill_function : {'mean', 'median'}

        Returns
        -------

        """
        if values is not None:
            if method is not None:
                raise ValueError('You cannot specify both `values` and and a `method` '
                                 'at the same time')
            if fill_function is not None:
                raise ValueError('You cannot specify both `values` and `fill_function` '
                                 'at the same time')
        if limit is not None:
            if not isinstance(limit, int) or limit < 1:
                raise ValueError('`limit` must be a positive integer')
        else:
            limit = len(self)

        if isinstance(values, (int, float, np.number)) and not isinstance(values, np.timedelta64):
            if self._is_string():
                raise TypeError("Your DataFrame contains only str columns and you are "
                                "trying to pass a number to fill in missing values")
            if self._is_date():
                raise TypeError("Your DataFrame contains only datetime/timedelta columns "
                                "and you are trying to pass a number to fill in missing values")

            new_data: Dict[str, ndarray] = {}
            for dtype, arr in self._data.items():
                arr = arr.copy('F')
                if dtype == 'f':
                    if limit >= len(self):
                        new_data['f'] = np.where(np.isnan(arr), values, arr)
                    else:
                        for col in self._columns:
                            dtype2, loc, _ = self._column_info[col].values
                            if dtype2 == 'f':
                                col_arr = arr[:, loc]
                                idx: ndarray = np.where(np.isnan(col_arr))[0][:limit]
                                # the following operation is a view of arr
                                col_arr[idx] = values
                        new_data['f'] = arr
                else:
                    new_data[dtype] = arr
        elif isinstance(values, str):
            if 'S' not in self._data:
                raise TypeError("You passed a `str` value to the `values` parameter. "
                                "You're DataFrame contains no str columns.")
            new_data = {}
            for dtype, arr in self._data.items():
                arr = arr.copy('F')
                if dtype == 'S':
                    for col in self._columns:
                        dtype2, loc, _ = self._column_info[col].values
                        if dtype2 == 'S':
                            col_arr = arr[:, loc]
                            na_arr: ndarray = _math.isna_str_1d(col_arr)
                            idx = np.where(na_arr)[0][:limit]
                            col_arr[idx] = values

                    new_data['S'] = arr
                else:
                    new_data[dtype] = arr
        elif isinstance(values, np.datetime64):
            if 'M' not in self._data:
                raise TypeError('You passed a `datetime64` value to the `values` parameter but '
                                'your DataFrame contains no datetime64 columns')
            new_data = {}
            for dtype, arr in self._data.items():
                arr = arr.copy('F')
                if dtype == 'M':
                    for col in self._columns:
                        dtype2, loc, _ = self._column_info[col].values
                        if dtype2 == 'M':
                            col_arr = arr[:, loc]
                            na_arr: ndarray = np.isnat(col_arr)
                            idx = np.where(na_arr)[0][:limit]
                            col_arr[idx] = values.astype('datetime64[ns]')

                    new_data['M'] = arr
                else:
                    new_data[dtype] = arr
        elif isinstance(values, np.timedelta64):
            if 'm' not in self._data:
                raise TypeError('You passed a `timedelta64` value to the `values` parameter but '
                                'your DataFrame contains no timedelta64 columns')
            new_data = {}
            for dtype, arr in self._data.items():
                arr = arr.copy('F')
                if dtype == 'm':
                    for col in self._columns:
                        dtype2, loc, _ = self._column_info[col].values
                        if dtype2 == 'm':
                            col_arr = arr[:, loc]
                            na_arr: ndarray = np.isnat(col_arr)
                            idx = np.where(na_arr)[0][:limit]
                            col_arr[idx] = values.astype('timedelta64[ns]')

                    new_data['m'] = arr
                else:
                    new_data[dtype] = arr
        elif isinstance(values, dict):
            self._validate_column_name_list(list(values))
            dtype_locs: Dict[str, List[Tuple[str, int, Scalar]]] = defaultdict(list)
            for col, val in values.items():
                dtype, loc = self._get_col_dtype_loc(col)  # type: str, int
                if dtype in 'fO':
                    dtype_locs[dtype].append((col, loc, val))

            for col, loc, new_val in dtype_locs['f']:
                if not isinstance(new_val, (int, float, np.number)):
                    raise TypeError(f'Column {col} has dtype float. Must set with a number')
            for col, loc, new_val in dtype_locs['S']:
                if not isinstance(new_val, str):
                    raise TypeError(f'Column {col} has dtype {dtype}. Must set with a str')

            arr_float: ndarray = self._data.get('f', []).copy('F')
            arr_str: ndarray = self._data.get('S', []).copy('F')
            for col, loc, new_val in dtype_locs['f']:
                if limit >= len(self):
                    arr_float[:, loc] = np.where(np.isnan(arr_float[:, loc]), new_val,
                                                 arr_float[:, loc])
                else:
                    idx = np.where(np.isnan(arr_float[:, loc]))[0][:limit]
                    arr_float[idx, loc] = new_val

            for col, loc, new_val in dtype_locs['S']:
                na_arr = _math.isna_str_1d(arr_str[:, loc])
                if limit >= len(self):
                    arr_str[:, loc] = np.where(na_arr, new_val, arr_str[:, loc])
                else:
                    idx = np.where(na_arr)[0][:limit]
                    arr_float[idx, loc] = new_val

            new_data = {}
            for dtype, arr in self._data.items():
                if dtype == 'f':
                    new_data[dtype] = arr_float
                elif dtype == 'S':
                    new_data[dtype] = arr_str
                else:
                    new_data[dtype] = arr.copy("F")

        elif values is None:
            if method is None and fill_function is None:
                raise ValueError('One of `values`, `method`, or `fill_function` must not be None')
            if method is not None:
                if fill_function is not None:
                    raise ValueError('You cannot specify both `method` and `fill_function` '
                                     'at the same time')
                if method == 'ffill':
                    new_data = {}
                    for dtype, arr in self._data.items():
                        arr = arr.copy('F')
                        if dtype == 'f':
                            new_data['f'] = _math.ffill_float(arr, limit)
                        elif dtype == 'S':
                            new_data['S'] = _math.ffill_str(arr, limit)
                        elif dtype in 'mM':
                            nans = np.isnat(arr)
                            dtype_name = 'datetime64[ns]' if dtype == 'M' else 'timedelta64[ns]'
                            new_data[dtype] = _math.ffill_date(arr.view('int64'), limit,
                                                               nans).astype(dtype_name)
                        else:
                            new_data[dtype] = arr
                elif method == 'bfill':
                    new_data = {}
                    for dtype, arr in self._data.items():
                        arr = arr.copy('F')
                        if dtype == 'f':
                            new_data['f'] = _math.bfill_float(arr, limit)
                        elif dtype == 'S':
                            new_data['S'] = _math.bfill_str(arr, limit)
                        elif dtype in 'mM':
                            nans = np.isnat(arr)
                            dtype_name = 'datetime64[ns]' if dtype == 'M' else 'timedelta64[ns]'
                            new_data[dtype] = _math.bfill_date(arr.view('int64'), limit,
                                                               nans).astype(dtype_name)
                        else:
                            new_data[dtype] = arr
                else:
                    raise ValueError('`method` must be either "bfill" or "ffill"')
            else:
                # fill_function must be not none
                # TODO make limit work with fill function
                if fill_function not in ['mean', 'median']:
                    raise ValueError('`fill_function` must be either "mean" or "median"')
                new_data = {}
                for dtype, arr in self._data.items():
                    arr = arr.copy('F')
                    if dtype == 'f':
                        fill_vals = getattr(self, fill_function)().values
                        new_data['f'] = np.where(np.isnan(arr), fill_vals, arr)
                    else:
                        new_data[dtype] = arr
        else:
            raise TypeError('`values` must be either an int, str, dict or None. You passed '
                            f'{values}')

        new_columns = self._columns.copy()
        new_column_info = self._copy_column_info()
        return self._construct_from_new(new_data, new_column_info, new_columns)

    def _replace_nans(self, dtype: str, col_arr: ndarray, asc: bool, hasnans: ndarray,
                      return_na_arr: bool = False):
        if dtype == 'S':
            if hasnans or hasnans is None:
                if asc:
                    nan_value = chr(10 ** 6)
                else:
                    nan_value = ''
                if col_arr.ndim == 1:
                    na_arr: ndarray = _math.isna_str_1d(col_arr)
                    arr_final: ndarray = np.where(na_arr, nan_value, col_arr)
                else:
                    hasnans = np.array([True] * col_arr.shape[1])  # TODO: use np.fill
                    na_arr = _math.isna_str(col_arr, hasnans)
                    arr_final = np.where(na_arr, nan_value, col_arr)
        elif dtype == 'f':
            if hasnans or hasnans is None:
                if asc:
                    nan_value = np.inf
                else:
                    nan_value = -np.inf
                # TODO: check individual columns for nans for speed increase
                na_arr = np.isnan(col_arr)
                arr_final = np.where(na_arr, nan_value, col_arr)
        elif dtype in 'mM':
            if hasnans or hasnans is None:
                if asc:
                    if dtype == 'M':
                        nan_value = np.datetime64(np.iinfo('int64').max, 'ns')
                    else:
                        nan_value = np.timedelta64(np.iinfo('int64').max, 'ns')
                else:
                    if dtype == 'M':
                        nan_value = np.datetime64(np.iinfo('int64').min, 'ns')
                    else:
                        nan_value = np.timedelta64(np.iinfo('int64').min, 'ns')
                # TODO: check individual columns for nans for speed increase
                na_arr = np.isnat(col_arr)
                arr_final = np.where(na_arr, nan_value, col_arr)
        else:
            return col_arr

        if return_na_arr:
            return arr_final, na_arr
        else:
            return arr_final

    def sort_values(self, by: Union[str, List[str]], axis: str = 'rows',
                    ascending: Union[bool, List[bool]] = True) -> 'DataFrame':
        axis_num = utils.convert_axis_string(axis)
        if axis_num == 1:
            raise NotImplementedError('Not implemented for sorting rows')
        if isinstance(by, str):
            self._validate_column_name(by)
            by = [by]
        elif isinstance(by, list):
            self._validate_column_name_list(by)
        else:
            raise TypeError('`by` variable must either be a column name as a string or a '
                            'list of column names as strings')

        if isinstance(ascending, list):
            if len(ascending) != len(by):
                raise ValueError('The number of columns in `by` does not match the number of '
                                 f'booleans in `ascending` list {len(by)} != {len(ascending)}')
            for asc in ascending:
                if not isinstance(asc, bool):
                    raise TypeError('All values passed to `ascending` list must be boolean')
        elif not isinstance(ascending, bool):
            raise TypeError('`ascending` must be a boolean or list of booleans')
        else:
            ascending = [ascending] * len(by)

        if len(by) == 1:
            col = by[0]
            dtype, loc = self._get_col_dtype_loc(col)  # type: str, int
            col_arr = self._data[dtype][:, loc]
            hasnans: ndarray = self._hasnans.get(col, True)
            asc = ascending[0]
            col_arr = self._replace_nans(dtype, col_arr, asc, hasnans)
            count_sort: bool = False
            if dtype == 'S':
                if len(col_arr) > 1000 and len(set(np.random.choice(col_arr, 100))) <= 70:
                    d: ndarray = _sr.sort_str_map(col_arr, asc)
                    arr: ndarray = _sr.replace_str_int(col_arr, d)
                    counts: ndarray = _sr.count_int_ordered(arr, len(d))
                    new_order: ndarray = _sr.get_idx(arr, counts)
                    count_sort = True
                else:
                    col_arr = col_arr.astype('U')
                    if asc:
                        new_order = np.argsort(col_arr, kind='mergesort')
                    else:
                        new_order = np.argsort(col_arr[::-1], kind='mergesort')
            else:
                if asc:
                    new_order = np.argsort(col_arr, kind='mergesort')
                else:
                    new_order = np.argsort(col_arr[::-1], kind='mergesort')

            new_data: Dict[str, ndarray] = {}
            for dtype, arr in self._data.items():
                np_dtype = utils.convert_kind_to_numpy(dtype)
                arr_final = np.empty(arr.shape, dtype=np_dtype, order='F')
                for i in range(arr.shape[1]):
                    if asc or count_sort:
                        arr_final[:, i] = arr[:, i][new_order]
                    else:
                        arr_final[:, i] = arr[::-1, i][new_order[::-1]]
                new_data[dtype] = arr_final
            new_column_info = self._copy_column_info()
            new_columns = self._columns.copy()

            return self._construct_from_new(new_data, new_column_info, new_columns)
        else:
            single_cols: List[ndarray] = []
            for col, asc in zip(by, ascending):
                dtype, loc = self._get_col_dtype_loc(col)  # type: str, int
                col_arr = self._data[dtype][:, loc]
                hasnans = self._hasnans.get(col, True)
                col_arr = self._replace_nans(dtype, col_arr, asc, hasnans)

                if not asc:
                    if dtype == 'b':
                        col_arr = ~col_arr
                    elif dtype == 'S':
                        # TODO: how to avoid mapping to ints for mostly unique string columns?
                        d = _sr.sort_str_map(col_arr, asc)
                        col_arr = _sr.replace_str_int(col_arr, d)
                    elif dtype == 'M':
                        col_arr = (-(col_arr.view('int64') + 1)).astype('datetime64[ns]')
                    elif dtype == 'm':
                        col_arr = (-(col_arr.view('int64') + 1)).astype('timedelta64[ns]')
                    else:
                        col_arr = -col_arr
                elif dtype == 'S':
                    if len(col_arr) > 1000 and len(set(np.random.choice(col_arr, 100))) <= 70:
                        d = _sr.sort_str_map(col_arr, asc)
                        col_arr = _sr.replace_str_int(col_arr, d)
                    else:
                        col_arr = col_arr.astype('U')

                single_cols.append(col_arr)

            new_order = np.lexsort(single_cols[::-1])

        new_data = {}
        for dtype, arr in self._data.items():
            np_dtype = utils.convert_kind_to_numpy(dtype)
            arr_final = np.empty(arr.shape, dtype=np_dtype, order='F')
            for i in range(arr.shape[1]):
                arr_final[:, i] = arr[:, i][new_order]
            new_data[dtype] = arr_final
        new_column_info = self._copy_column_info()
        new_columns = self._columns.copy()

        return self._construct_from_new(new_data, new_column_info, new_columns)

    def _get_all_dtype_info(self) -> Tuple[Dict[str, List[str]],
                                           Dict[str, List[int]],
                                           Dict[str, List[int]]]:

        dtype_col: Dict[str, List[str]] = defaultdict(list)
        dtype_loc: Dict[str, List[int]] = defaultdict(list)
        dtype_order: Dict[str, List[int]] = defaultdict(list)

        for col, dtype, loc, order in self._col_info_iter(with_order=True):  # type: str, str, int, int
            dtype_order[dtype].append(order)
            dtype_loc[dtype].append(loc)
            dtype_col[dtype].append(col)
        return dtype_col, dtype_loc, dtype_order

    def _get_all_dtype_info_subset(self, subset: Optional[List[str]] = None) -> Tuple[
        Dict[str, List[str]], Dict[str, List[int]], ndarray, Dict[str, int]]:

        dtype_col: Dict[str, List[str]] = defaultdict(list)
        dtype_loc: Dict[str, List[int]] = defaultdict(list)
        new_col_order: Dict[str, int] = {}

        if subset is None or len(subset) == len(self._columns):
            for i, (col, dtype, loc) in enumerate(self._col_info_iter()):  # type: str, str, int
                dtype_loc[dtype].append(loc)
                dtype_col[dtype].append(col)
                new_col_order[col] = i

            new_columns = self._columns.copy()
        else:
            col_set = set(subset)
            new_columns = np.empty(len(col_set), dtype='O')
            i = 0
            for col, dtype, loc in self._col_info_iter():  # type: str, str, int
                if col not in col_set:
                    continue
                new_columns[i] = col
                new_col_order[col] = i
                i += 1
                dtype_loc[dtype].append(loc)
                dtype_col[dtype].append(col)

        return dtype_col, dtype_loc, new_columns, new_col_order

    def rank(self, axis: str = 'rows', method: str = 'min', na_option: str = 'keep',
             ascending: bool = True) -> 'DataFrame':
        axis_num = utils.convert_axis_string(axis)
        if axis_num == 1:
            raise NotImplementedError('Can only rank columns for now :(')
        if method not in ('min', 'max', 'dense', 'first', 'average'):
            raise ValueError("`method` must be either 'min', 'max', 'dense', 'first', or 'average'")
        if na_option in ('keep', 'bottom'):
            if ascending:
                na_asc = True
            else:
                na_asc = False
        elif na_option == 'top':
            if ascending:
                na_asc = False
            else:
                na_asc = True
        else:
            raise ValueError("`na_option must be 'keep', 'top', or 'bottom'")

        def get_cur_rank(dtype: str, arr: ndarray) -> ndarray:
            if na_option == 'keep' and dtype in 'fOmM':
                arr, na_arr = self._replace_nans(dtype, arr, na_asc, True, True)
            else:
                arr = self._replace_nans(dtype, arr, na_asc, True)

            func_name: str = 'rank_' + utils.convert_kind_to_dtype_generic(dtype) + '_' + method

            if not ascending:
                if dtype == 'S':
                    arg = np.argsort(arr, 0, kind='mergesort')[::-1]
                elif dtype == 'b':
                    arg = np.argsort(~arr, 0, kind='mergesort')
                elif dtype in 'mM':
                    # np.nat evaluates as min possible int
                    arg = np.argsort(-(arr.view('int64') + 1), 0, kind='mergesort')
                else:
                    arg = np.argsort(-arr, 0, kind='mergesort')
            else:
                arg = np.argsort(arr, 0, kind='mergesort')

            if method == 'first' and dtype == 'S' and not ascending:
                cur_rank = _sr.rank_str_min(arg, arr)
                cur_rank = _sr.rank_str_min_to_first(cur_rank, arg, arr)
            else:
                if dtype in 'mM':
                    arr = arr.view('int64')
                cur_rank = getattr(_sr, func_name)(arg, arr)

            if na_option == 'keep' and dtype in 'fOmM' and na_arr.sum() > 0:
                # TODO: only convert columns with nans to floats
                cur_rank = cur_rank.astype('float64')
                cur_rank[na_arr] = nan
            return cur_rank

        dtype_col, dtype_loc, dtype_order = self._get_all_dtype_info()
        data_dict: DictListArr = defaultdict(list)
        new_column_info: ColInfoT = {}
        for dtype, arr in self._data.items():
            cur_rank = get_cur_rank(dtype, arr)
            new_dtype = cur_rank.dtype.kind
            cur_loc = utils.get_num_cols(data_dict.get(new_dtype, []))
            data_dict[new_dtype].append(cur_rank)

            for col, loc, order in zip(dtype_col[dtype], dtype_loc[dtype], dtype_order[dtype]):
                new_column_info[col] = utils.Column(new_dtype, loc + cur_loc, order)

        new_data = utils.concat_stat_arrays(data_dict)
        new_columns = self._columns.copy()
        return self._construct_from_new(new_data, new_column_info, new_columns)

    def value_counts(self, col: str, normalize: bool = False, sort: bool = True,
                     dropna: bool = True) -> 'DataFrame':
        if not isinstance(col, str):
            raise TypeError('`col` must be the name of a column')
        self._validate_column_name(col)
        dtype, loc = self._get_col_dtype_loc(col)  # type: str, int
        arr = self._data[dtype][:, loc]
        if dtype == 'S':
            groups, counts = _gb.value_counts_str(arr, dropna=dropna)
            uniques = arr[groups]
        elif dtype == 'i':
            low, high = _math.min_max_int(arr)
            if high - low < 10_000_000:
                uniques, counts = _gb.value_counts_int_bounded(arr, low, high)
            else:
                groups, counts = _gb.value_counts_int(arr)
                uniques = arr[groups]
        elif dtype == 'f':
            groups, counts = _gb.value_counts_float(arr, dropna=dropna)
            uniques = arr[groups]
        elif dtype == 'b':
            uniques, counts = _gb.value_counts_bool(arr)
        elif dtype in 'mM':
            groups, counts = _gb.value_counts_int(arr.view('int64'))
            uniques = arr[groups]
            if dropna:
                keep = ~np.isnat(uniques)
                counts = counts[keep]
                uniques = uniques[keep]

        if normalize:
            counts = counts / counts.sum()
            counts_kind = 'f'
        else:
            counts_kind = 'i'

        if sort:
            order = np.argsort(counts)[::-1]
            uniques = uniques[order]
            counts = counts[order]

        unique_kind = uniques.dtype.kind
        if unique_kind == counts_kind:
            new_data = {counts_kind: np.asfortranarray(np.column_stack((uniques, counts)))}
            new_column_info = {col: utils.Column(unique_kind, 0, 0),
                               'count': utils.Column(counts_kind, 1, 1)}
        else:
            new_data = {unique_kind: uniques[:, np.newaxis], counts_kind: counts[:, np.newaxis]}
            new_column_info = {col: utils.Column(unique_kind, 0, 0),
                               'count': utils.Column(counts_kind, 0, 1)}

        new_columns = np.array([col, 'count'], dtype='O')

        return self._construct_from_new(new_data, new_column_info, new_columns)

    def groupby(self, columns: Union[str, List[str]]) -> 'Grouper':
        from ._groupby import Grouper

        if isinstance(columns, list):
            self._validate_column_name_list(columns)
        elif isinstance(columns, str):
            self._validate_column_name(columns)
            columns = [columns]
        else:
            raise ValueError('Must pass in grouping column(s) as a string or list of strings')
        return Grouper(self, columns)

    def streak(self, column: Optional[str] = None, value: Optional[Scalar] = None,
               group: bool = False) -> ndarray:
        """
        Three types of streaks for a single column. Must specify Column
        All values - begin at 1, value=None
        Specific value given by value
        group - each group is given same number
        """
        if not isinstance(column, str):
            raise TypeError('`column` must be a column name as a string')
        else:
            self._validate_column_name(column)

        if not isinstance(group, (bool, np.bool_)):
            raise TypeError('`group` must be either True or False')

        dtype, loc = self._get_col_dtype_loc(column)  # type: str, int
        col_arr = self._data[dtype][:, loc]

        if not group:
            if value is None:
                func_name = 'streak_' + utils.convert_kind_to_dtype_generic(dtype)
                func = getattr(_math, func_name)
                if dtype in 'mM':
                    return func(col_arr.view('int64'))
                else:
                    return func(col_arr)
            else:
                if dtype == 'i':
                    if not isinstance(value, (int, np.integer)):
                        raise TypeError(f'Column {column} has dtype int and `value` is a '
                                        f'{type(value).__name__}.')
                    return _math.streak_value_int(col_arr, value)
                elif dtype == 'f':
                    if not isinstance(value, (int, float, np.number)):
                        raise TypeError(f'Column {column} has dtype float and `value` is a '
                                        f'{type(value).__name__}.')
                    return _math.streak_value_float(col_arr, value)
                elif dtype == 'S':
                    if not isinstance(value, str):
                        raise TypeError(f'Column {column} has dtype str and `value` is a '
                                        f'{type(value).__name__}.')
                    return _math.streak_value_str(col_arr, value)
                elif dtype == 'b':
                    if not isinstance(value, (bool, np.bool_)):
                        raise TypeError(f'Column {column} has dtype bool and `value` is a '
                                        f'{type(value).__name__}.')
                    return _math.streak_value_bool(col_arr, value)
                elif dtype == 'M':
                    if not isinstance(value, np.datetime64):
                        raise TypeError(f'Column {column} has dtype datetime64 and `value` is a '
                                        f'{type(value).__name__}.')
                    return _math.streak_value_int(col_arr.view('int64'), value.astype('int64'))
                elif dtype == 'm':
                    if not isinstance(value, np.timedelta64):
                        raise TypeError(f'Column {column} has dtype timedelta64 and `value` is a '
                                        f'{type(value).__name__}.')
                    value = value.astype('timedelta64[ns]').astype('int64')
                    return _math.streak_value_int(col_arr.view('int64'), value)
        else:
            if value is not None:
                raise ValueError('If `group` is True then `value` must be None')

            func_name = 'streak_group_' + utils.convert_kind_to_dtype_generic(dtype)
            func = getattr(_math, func_name)
            if dtype in 'mM':
                col_arr = col_arr.view('int64')
            return func(col_arr)

    def rename(self, columns: Union[str, List[str]]) -> 'DataFrame':
        """
        pass in a list to rename all columns
        use a dictionary to rename specific columns
        or a callable

        """
        if isinstance(columns, (list, ndarray)):
            new_columns = init.check_column_validity(columns)

        elif isinstance(columns, dict):
            for col in columns:
                if col not in self._column_info:
                    raise ValueError(f'Column {col} is not a column')

            new_columns = [columns.get(col, col) for col in self._columns]
            new_columns = init.check_column_validity(new_columns)

        elif callable(columns):
            new_columns = [columns(col) for col in self._columns]
            new_columns = init.check_column_validity(new_columns)
        else:
            raise ValueError('You must pass either a dictionary, list/array, '
                             'or function to `columns`')

        if len(new_columns) != len(self._columns):
            raise ValueError('The number of strings in your list/array of new columns '
                             'does not match the number of columns '
                             f'{len(new_columns)} ! {len(self._columns)}')

        new_column_info = {}
        for old_col, new_col in zip(self._columns, new_columns):
            new_column_info[new_col] = utils.Column(*self._column_info[old_col].values)

        new_data = {}
        for dtype, arr in self._data.items():
            new_data[dtype] = arr.copy('F')

        return self._construct_from_new(new_data, new_column_info,
                                        np.asarray(new_columns, dtype='O'))

    def _drop_just_cols(self, columns):
        if isinstance(columns, (int, str, np.integer)):
            columns = [columns]
        elif isinstance(columns, ndarray):
            columns = utils.try_to_squeeze_array(columns)
        elif isinstance(columns, list):
            pass
        elif columns is not None:
            raise TypeError('Columns must either be an int, list/array of ints or None')
        else:
            raise ValueError('Both rows and columns cannot be None')

        column_strings: Set[str] = set()
        for col in columns:
            if isinstance(col, str):
                column_strings.add(col)
            elif isinstance(col, (int, np.integer)):
                column_strings.add(self._columns[col])

        dtype_drop_info = defaultdict(list)
        for col in column_strings:
            dtype, loc = self._get_col_dtype_loc(col)  # type: str, int
            dtype_drop_info[dtype].append(loc)

        new_column_info = {}
        new_columns = []
        order_sub = 0
        for col, dtype, loc, order in self._col_info_iter(with_order=True):  # type: str, str, int, int
            if col not in column_strings:
                loc_sub = 0
                for loc_drops in dtype_drop_info.get(dtype, ()):
                    loc_sub += loc > loc_drops
                new_column_info[col] = utils.Column(dtype, loc - loc_sub, order - order_sub)
                new_columns.append(col)
            else:
                order_sub += 1

        new_data = {}
        for dtype, arr in self._data.items():
            if dtype not in dtype_drop_info:
                new_data[dtype] = arr.copy('F')
            else:
                locs = dtype_drop_info[dtype]
                if locs == arr.shape[1]:
                    continue
                keep = np.ones(arr.shape[1], dtype='bool')
                keep[locs] = False
                new_data[dtype] = arr[:, keep]

        return self._construct_from_new(new_data, new_column_info,
                                        np.asarray(new_columns, dtype='O'))

    def _drop_just_rows(self, rows):
        if isinstance(rows, int):
            rows = [rows]
        elif isinstance(rows, ndarray):
            rows = utils.try_to_squeeze_array(rows)
        elif isinstance(rows, list):
            pass
        else:
            raise TypeError('Rows must either be an int, list/array of ints or None')
        new_data = {}
        for dtype, arr in self._data.items():
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                try:
                    new_data[dtype] = np.delete(arr, rows, axis=0)
                except (DeprecationWarning, FutureWarning):
                    raise IndexError('one of the rows is out of bounds')
        new_column_info = self._copy_column_info()
        new_columns = self._columns.copy()
        return self._construct_from_new(new_data, new_column_info, new_columns)

    def drop(self,
             rows: Union[int, List[int], ndarray, None] = None,
             columns: Union[str, int, List[IntStr], ndarray, None] = None):
        if rows is None:
            return self._drop_just_cols(columns)

        if columns is None:
            return self._drop_just_rows(rows)

        if isinstance(columns, (int, str, np.integer)):
            columns = [columns]
        elif isinstance(columns, ndarray):
            columns = utils.try_to_squeeze_array(columns)
        elif not isinstance(columns, list):
            raise TypeError('Rows must either be an int, list/array of ints or None')

        if isinstance(rows, int):
            rows = [rows]
        elif isinstance(rows, ndarray):
            rows = utils.try_to_squeeze_array(rows)
        elif not isinstance(rows, list):
            raise TypeError('Rows must either be an int, list/array of ints or None')

        new_rows: List[int] = []
        for row in rows:
            if not isinstance(row, int):
                raise TypeError('All the row values in your list must be integers')
            if row < -len(self) or row >= len(self):
                raise IndexError(f'Integer location {row} for the rows is out of range')
            if row < 0:
                new_rows.append(len(self) + row)
            else:
                new_rows.append(row)

        column_strings: List[str] = []
        for col in columns:
            if isinstance(col, str):
                column_strings.append(col)
            elif isinstance(col, (int, np.integer)):
                column_strings.append(self._columns[col])

        self._validate_column_name_list(column_strings)
        column_set: Set[str] = set(column_strings)

        new_rows = np.isin(np.arange(len(self)), new_rows, invert=True)
        new_columns = [col for col in self._columns if col not in column_set]

        new_column_info: ColInfoT = {}
        data_dict: Dict[str, List[int]] = defaultdict(list)
        for i, col in enumerate(new_columns):
            dtype, loc = self._get_col_dtype_loc(col)  # type: str, int
            cur_loc = len(data_dict[dtype])
            new_column_info[col] = utils.Column(dtype, cur_loc, i)
            data_dict[dtype].append(loc)

        new_data = {}
        for dtype, locs in data_dict.items():
            new_data[dtype] = self._data[dtype][np.ix_(new_rows, locs)]

        return self._construct_from_new(new_data, new_column_info,
                                        np.asarray(new_columns, dtype='O'))

    def _nest(self, n: int, column: str, keep: str, name: str) -> 'DataFrame':
        if not isinstance(n, (int, np.integer)):
            raise TypeError('`n` must be an integer')
        if n < 1:
            raise ValueError('`n` must be a positive integer')
        if keep not in ['first', 'last', 'all']:
            raise ValueError('`keep` must be either "all", "first", or "last"')

        if not isinstance(column, str):
            raise TypeError('`column` must a name of a column as a string')

        self._validate_column_name(column)
        dtype, loc = self._get_col_dtype_loc(column)  # type: str, int
        col_arr = self._data[dtype][:, loc]

        if n < 100:
            if dtype in 'mM':
                dtype = 'f'
                na_arr = np.isnat(col_arr)
                col_arr = col_arr.astype('float64')
                col_arr[na_arr] = nan
            func_name = name + '_' + utils.convert_kind_to_dtype(dtype)
            order, ties = getattr(_math, func_name)(col_arr, n)
            if keep == 'all' and ties:
                order = np.append(order, ties)
            elif keep == 'last':
                if ties:
                    tie_val = col_arr[ties[0]]
                    is_ties = col_arr[order] == tie_val
                    num_ties = is_ties.sum()
                    if len(ties) >= num_ties:
                        order[is_ties] = ties[-num_ties:]
                    else:
                        order[-num_ties:] = np.append(order[is_ties], ties)[-num_ties:]
        else:
            if dtype in 'mM':
                func_name = 'quick_select_int'
            else:
                func_name = 'quick_select_' + utils.convert_kind_to_dtype(dtype)

            if col_arr.dtype.kind == 'f':
                not_na = ~np.isnan(col_arr)
                col_arr_final = col_arr[not_na]
            elif col_arr.dtype.kind == 'S':
                not_na = ~_math.isna_str_1d(col_arr)
                col_arr_final = col_arr[not_na]
            elif col_arr.dtype.kind == 'b':
                col_arr_final = col_arr.astype('int64')
            elif col_arr.dtype.kind in 'mM':
                not_na = ~np.isnat(col_arr)
                col_arr = col_arr.view('int64')
                col_arr_final = col_arr[not_na]
            else:
                col_arr_final = col_arr

            if dtype not in 'mM':
                dtype = col_arr_final.dtype.kind

            if n > len(col_arr):
                asc = name == 'nsmallest'
                return self.sort_values(column, ascending=asc)

            if name == 'nlargest':
                if dtype == 'S':
                    nth = _math.quick_select_str(col_arr_final, len(col_arr_final) - n)
                    col_arr_final = _va.fill_str_none(col_arr, False)
                    idx = np.where((col_arr_final >= nth) & not_na)[0]
                    vals = col_arr_final[idx]
                    idx_args = (len(vals) - 1 - np.argsort(vals[::-1], kind='mergesort'))[::-1]
                else:
                    nth = -getattr(_math, func_name)(-col_arr_final, n)
                    if dtype in 'fmM':
                        col_arr_final = col_arr
                    with np.errstate(invalid='ignore'):
                        idx = np.where(col_arr_final >= nth)[0]
                    vals = col_arr_final[idx]
                    idx_args = np.argsort(-vals, kind='mergesort')
            else:
                nth = getattr(_math, func_name)(col_arr_final, n)
                if dtype == 'S':
                    col_arr_final = _va.fill_str_none(col_arr, False)
                    idx = np.where(col_arr_final >= nth & not_na)[0]
                else:
                    if dtype in 'fmM':
                        col_arr_final = col_arr
                    idx = np.where(col_arr_final >= nth)[0]
                vals = col_arr_final[idx]
                idx_args = np.argsort(vals, kind='mergesort')

            if keep == 'first':
                idx_args = idx_args[:n]
            elif keep == 'last' and len(idx_args) > n:
                tie_args = idx_args[n:]
                idx_args = idx_args[:n]
                tie_val = vals[tie_args[-1]]
                is_ties = vals[idx_args] == tie_val
                num_ties = is_ties.sum()

                if len(tie_args) >= num_ties:
                    idx_args[is_ties] = tie_args[-num_ties:]
                else:
                    idx_args[-num_ties:] = np.append(idx_args[is_ties], tie_args)[-num_ties:]

            order = idx[idx_args]

        new_column_info = self._copy_column_info()
        new_columns = self._columns.copy()
        new_data = {}
        for dtype, arr in self._data.items():
            new_data[dtype] = arr[order]

        return self._construct_from_new(new_data, new_column_info, new_columns)

    def nlargest(self, n: int, column: str, keep: str = 'all'):
        return self._nest(n, column, keep, 'nlargest')

    def nsmallest(self, n: int, column: str, keep: str = 'all'):
        return self._nest(n, column, keep, 'nsmallest')

    def factorize(self, column: str) -> Tuple[ndarray, ndarray]:
        self._validate_column_name(column)
        dtype, loc = self._get_col_dtype_loc(column)  # type: str, int
        col_arr = self._data[dtype][:, loc]
        if dtype in 'mM':
            col_arr = col_arr.view('int64')
            dtype = 'i'
        func_name = 'get_group_assignment_' + utils.convert_kind_to_dtype(dtype) + '_1d'
        groups, first_pos = getattr(_gb, func_name)(col_arr)
        return groups, col_arr[first_pos]

    def sample(self, n: Optional[int] = None, frac: Optional[float] = None, replace: bool = False,
               weights: Union[List, ndarray, None] = None,
               random_state: Optional[Any] = None,
               axis: str = 'rows') -> 'DataFrame':
        axis_num = utils.convert_axis_string(axis)
        if axis_num == 1:
            raise NotImplementedError('No sampling columns yet')

        if not isinstance(replace, (bool, np.bool_)):
            raise TypeError('`replace` must be either True or False')

        if weights is not None:
            if axis_num == 0 and len(weights) != len(self):
                raise ValueError('`weights` must have the same number of elements as the '
                                 'number of rows in the DataFrame ')
            if axis_num == 1 and len(weights) != len(self._columns):
                raise ValueError('`weights` must have the same number of elements as the '
                                 'number of columns in the DataFrame ')
            weights = np.asarray(weights)
            weights = utils.try_to_squeeze_array(weights)
            if weights.dtype.kind not in ('i', 'f'):
                raise TypeError('All values in `weights` must be numeric')

            weight_sum = weights.sum()
            if np.isnan(weight_sum):
                weights[np.isnan(weights)] = 0
                weight_sum = weights.sum()

            if weight_sum <= 0:
                raise ValueError('The sum of the weights is <= 0. ')
            weights = weights / weight_sum

        if random_state is not None:
            if isinstance(random_state, (int, np.integer)):
                random_state = np.random.RandomState(random_state)
            elif not isinstance(random_state, np.random.RandomState):
                raise TypeError('`random_state` must either be ')

        if axis_num == 0:
            axis_len = len(self)
        else:
            axis_len = len(self._columns)

        if n is not None:
            if not isinstance(n, (int, np.integer)):
                raise TypeError('`n` must be either an integer or None')
            if n < 1:
                raise ValueError('`n` must greater than 0')
            if frac is not None:
                raise ValueError('You cannot specify both `n` and `frac`. Choose one or the other.')
        else:
            if frac is None:
                raise ValueError('`n` and `frac` cannot both be None. One and only one must be set')
            if not isinstance(frac, (int, float, np.number)) or frac <= 0:
                raise ValueError('`frac` must be a number greater than 0')

            n = ceil(frac * axis_len)

        if random_state is None:
            if weights is None and replace:
                new_idx = np.random.randint(0, axis_len, n)
            else:
                new_idx = np.random.choice(np.arange(axis_len), n, replace, weights)
        else:
            new_idx = random_state.choice(np.arange(axis_len), n, replace, weights)

        if axis_num == 0:
            new_columns = self._columns.copy()
            new_column_info = self._copy_column_info()
            new_data = {}
            for dtype, arr in self._data.items():
                new_data[dtype] = arr[new_idx]
            return self._construct_from_new(new_data, new_column_info, new_columns)
        else:
            column_ints: Dict[str, int] = defaultdict(int)
            data_dict: DictListArr = defaultdict(list)
            new_columns = []
            new_column_info = {}
            for i, num in enumerate(new_idx):
                col = self._columns[num]
                dtype, loc = self._get_col_dtype_loc(col)  # type: str, int
                cur_col_num = column_ints[col]
                if cur_col_num == 0:
                    col_new = col
                else:
                    col_new = col + str(cur_col_num)
                column_ints[col] += 1
                new_column_info[col_new] = utils.Column(dtype, cur_col_num, i)
                data_dict[dtype].append(self._data[dtype][:, [loc]])
                new_columns.append(col_new)

            new_data = utils.concat_stat_arrays(data_dict)
            return self._construct_from_new(new_data, new_column_info,
                                            np.asarray(new_columns, dtype='O'))

    def isin(self, values: Union[Scalar, List[Scalar],
                                 Dict[str, Union[Scalar, List[Scalar]]]]) -> 'DataFrame':
        if utils.is_scalar(values):
            values = [values]  # type: ignore

        def separate_value_types(vals: List[Scalar]) -> Tuple[List[Union[float, int, bool]], List[str]]:
            val_numbers = []
            val_strings = []
            val_datetimes = []
            val_timedeltas = []
            for val in vals:
                if isinstance(val, np.datetime64):
                    val_datetimes.append(val)
                elif isinstance(val, np.timedelta64):
                    val_timedeltas.append(val)
                elif isinstance(val, (float, int, np.number)):
                    val_numbers.append(val)
                elif isinstance(val, str):
                    val_strings.append(val)
            return val_numbers, val_strings, val_datetimes, val_timedeltas

        if isinstance(values, list):
            for value in values:
                if not utils.is_scalar(value):
                    raise ValueError('All values in list must be either int, float, str, or bool')
            arrs: List[ndarray] = []
            val_numbers, val_strings, val_datetimes, val_timedeltas = separate_value_types(values)
            dtype_add = {}
            for dtype, arr in self._data.items():
                dtype_add[dtype] = utils.get_num_cols(arrs)
                if dtype == 'S':
                    arrs.append(np.isin(arr, val_strings))
                elif dtype == 'M':
                    arrs.append(np.isin(arr, val_datetimes))
                elif dtype == 'm':
                    arrs.append(np.isin(arr, val_timedeltas))
                else:
                    arrs.append(np.isin(arr, val_numbers))

            new_column_info = {}
            for col, dtype, loc, order in self._col_info_iter(with_order=True):  # type: str, str, int, int
                new_column_info[col] = utils.Column('b', loc + dtype_add[dtype], order)

            new_columns = self._columns.copy()
            new_data = {'b': np.asfortranarray(np.column_stack(arrs))}
            return self._construct_from_new(new_data, new_column_info, new_columns)
        elif isinstance(values, dict):
            self._validate_column_name_list(list(values))
            arr_final = np.full(self.shape, False, dtype='bool')
            for col, vals in values.items():
                if utils.is_scalar(vals):
                    vals = [vals]  # type: ignore
                if not isinstance(vals, list):
                    raise TypeError('The dictionary values must be lists or a scalar')
                dtype, loc, order = self._get_col_dtype_loc_order(col)  # type: str, int, int
                col_arr = self._data[dtype][:, loc]
                val_numbers, val_strings, val_datetimes, val_timedeltas = separate_value_types(vals)
                if dtype == 'S':
                    arr_final[:, order] = np.isin(col_arr, val_strings)
                elif dtype == 'M':
                    arr_final[:, order] = np.isin(col_arr, val_datetimes)
                elif dtype == 'm':
                    arr_final[:, order] = np.isin(col_arr, val_timedeltas)
                else:
                    arr_final[:, order] = np.isin(col_arr, val_numbers)
            new_data = {'b': arr_final}
            new_columns = self._columns.copy()

            new_column_info = {}
            for i, col in enumerate(self._columns):
                new_column_info[col] = utils.Column('b', i, i)
            return self._construct_from_new(new_data, new_column_info, new_columns)
        else:
            raise TypeError("`values` must be a scalar, list, or dictionary of scalars/lists")

    def iterrows(self) -> Generator:
        for row in np.nditer(self.values, flags=['refs_ok', 'external_loop'], order='C'):
            yield row

    def __iter__(self) -> NoReturn:
        raise NotImplementedError('Use the `iterrows` method to iterate row by row. '
                                  'Manually write a `for` loop to iterate over each column.')

    def where(self, cond: Union[ndarray, 'DataFrame'],
              x: Union[Scalar, ndarray, 'DataFrame', None] = None,
              y: Union[Scalar, ndarray, 'DataFrame', None] = None) -> 'DataFrame':

        if isinstance(cond, DataFrame):
            cond = cond.values
        if isinstance(cond, ndarray):
            if cond.ndim == 1:
                cond = cond[:, np.newaxis]

            if cond.dtype.kind != 'b':
                raise TypeError('The `cond` numpy array must be boolean')
            if cond.shape[0] != self.shape[0]:
                raise ValueError('`cond` array must have the same number of rows as '
                                 f'calling DataFrame. {cond.shape[0]} != {self.shape[0]}')

            if cond.shape[1] != self.shape[1] and cond.shape[1] != 1:
                raise ValueError('`cond` must have either a single column or have the same number '
                                 'of columns as the calling DataFrame')
        else:
            raise TypeError('`cond` must be either a DataFrame or a NumPy array')

        if isinstance(x, ndarray) and x.ndim == 1:
            x = x[:, np.newaxis]

        if isinstance(y, ndarray) and y.ndim == 1:
            y = y[:, np.newaxis]

        if isinstance(x, DataFrame):
            x = x.values
        if isinstance(y, DataFrame):
            y = y.values

        for var, name in zip([x, y], ['x', 'y']):
            if isinstance(var, ndarray):
                if var.shape[0] != self.shape[0]:
                    raise ValueError(f'`{name} must have the same number of rows as the '
                                     'calling DataFrame')
                if var.shape[1] != self.shape[1] and var.shape[1] != 1:
                    raise ValueError(f'`{name}` must have either a single column '
                                     'or have the same number of columns of the calling DataFrame')

        def get_arr(arr: ndarray) -> ndarray:
            if arr.shape[1] == 1:
                return arr
            else:
                return arr[:, dtype_order[dtype]]

        def get_curr_var(var: Optional[ndarray], dtype: str, name: str) -> Optional[ndarray]:
            if var is None:
                return None if dtype == 'S' else nan

            types: Any
            if dtype == 'S':
                good_dtypes = ['S', 'U']
                types = str
            elif dtype in 'if':
                good_dtypes = ['i', 'f']
                types = (int, float, np.number)
            elif dtype == 'b':
                good_dtypes = ['b']
                types = (bool, np.bool_)
            elif dtype == 'M':
                good_dtypes = ['M']
                types = np.datetime64
            elif dtype == 'm':
                good_dtypes = ['m']
                types = np.timedelta64

            if isinstance(var, ndarray):
                var_curr = get_arr(var)

                if var_curr.dtype.kind not in good_dtypes:
                    raise TypeError(f'The array you passed to `{name}` must have compatible dtypes '
                                    'to the same columns in the calling DataFrame')
            elif not isinstance(var, types):
                raise TypeError(f'The values you are using to set with `{name}` do not '
                                'have compatible types with one of the columns')
            else:
                var_curr = var
            return var_curr

        # called when both x and y are given
        def check_x_y_types(x: Union[Scalar, ndarray, 'DataFrame', None],
                            y: Union[Scalar, ndarray, 'DataFrame', None]) -> Tuple[
            ndarray, ndarray]:
            if isinstance(x, ndarray):
                x = get_arr(x)

            if isinstance(y, ndarray):
                y = get_arr(y)

            if isinstance(x, ndarray) and isinstance(y, ndarray):
                if x.dtype.kind in 'if' and y.dtype.kind not in 'if':
                    raise TypeError('`x` and `y` arrays have incompatible dtypes. `x` is numeric '
                                    'and `y` is not')
                elif x.dtype.kind != y.dtype.kind:
                    raise TypeError('`x` and `y` dtypes are not compatible')

            elif isinstance(x, ndarray) and utils.is_scalar(y):
                if x.dtype.kind in 'if' and not isinstance(y, (int, float, np.number)):
                    raise TypeError('`x` and `y` arrays have incompatible dtypes. `x` is numeric '
                                    'and `y` is not')
                elif x.dtype.kind == 'S' and not isinstance(y, str):
                    raise TypeError('`x` and `y` arrays have incompatible dtypes. `x` is str '
                                    'and `y` is not')
                elif x.dtype.kind == 'b' and not isinstance(y, (bool, np.bool_)):
                    raise TypeError('`x` and `y` arrays have incompatible dtypes. `x` is bool '
                                    'and `y` is not')

            elif utils.is_scalar(x) and isinstance(y, ndarray):
                if y.dtype.kind in 'if' and not isinstance(x, (int, float, np.number)):
                    raise TypeError('`x` and `y` arrays have incompatible dtypes. `y` is numeric '
                                    'and `x` is not')
                elif y.dtype.kind == 'S' and not isinstance(x, str):
                    raise TypeError('`x` and `y` arrays have incompatible dtypes. `y` is str '
                                    'and `x` is not')
                elif y.dtype.kind == 'b' and not isinstance(x, (bool, np.bool_)):
                    raise TypeError('`x` and `y` arrays have incompatible dtypes. `y` is bool '
                                    'and `x` is not')
            elif utils.is_scalar(x) and utils.is_scalar(y):
                if (isinstance(x, (int, float, np.number)) and
                        not isinstance(y, (int, float, np.number))):
                    raise TypeError('`x` and `y` arrays have incompatible dtypes. `x` is numeric '
                                    'and `y` is not')
                elif isinstance(x, str) and not isinstance(y, str):
                    raise TypeError('`x` and `y` arrays have incompatible dtypes. `x` is str '
                                    'and `y` is not')
                elif isinstance(x, (bool, np.bool_)) and not isinstance(y, (bool, np.bool_)):
                    raise TypeError('`x` and `y` arrays have incompatible dtypes. `y` is bool '
                                    'and `x` is not')
            else:
                raise TypeError('`x` and `y` must be a scalar, array, DataFrame, or None')

            return x, y

        dtype_col, dtype_loc, dtype_order = self._get_all_dtype_info()

        data_dict: DictListArr = defaultdict(list)
        new_column_info = {}
        for dtype, arr in self._data.items():
            if cond.shape[1] != 1:
                cond_dtype = cond[:, dtype_order[dtype]]
            elif arr.shape[1] != 1:
                cond_dtype = cond[:, [0] * arr.shape[1]]
            else:
                cond_dtype = cond

            # dtype stays the same when object when x is None
            if x is None:
                x_curr = arr[:, dtype_loc[dtype]]
                y_curr = get_curr_var(y, dtype, 'y')
            elif y is None:
                x_curr = get_curr_var(x, dtype, 'x')
                y_curr = get_curr_var(y, dtype, 'y')
            else:
                x_curr, y_curr = check_x_y_types(x, y)

            arr_new = np.where(cond_dtype, x_curr, y_curr)
            if arr_new.dtype.kind == 'U':
                arr_new = arr_new.astype('O')
            new_dtype = arr_new.dtype.kind
            cur_loc = utils.get_num_cols(data_dict.get(new_dtype, []))
            data_dict[new_dtype].append(arr_new)

            for col, pos, order in zip(dtype_col[dtype], dtype_loc[dtype], dtype_order[dtype]):
                new_column_info[col] = utils.Column(new_dtype, pos + cur_loc, order)

        new_columns = self._columns.copy()
        new_data = utils.concat_stat_arrays(data_dict)
        return self._construct_from_new(new_data, new_column_info, new_columns)

    def rolling(self, left, right, min_window=None, kept_columns=False):
        if not isinstance(left, int):
            raise TypeError('`left` must be an integer')
        if not isinstance(right, int):
            raise TypeError('`right` must be an integer')
        if min_window is None:
            min_window = 0
        if not isinstance(min_window, int):
            raise TypeError('`min_window` must be either an integer or None')

        if left > right:
            raise ValueError('The `left` value must be less than or equal to the right')

        if min_window < 0 or min_window > right - left + 1:
            raise ValueError(f'`min_window` must be between 1 and {right - left + 1}')

        if not isinstance(kept_columns, (bool, str, list)):
            raise TypeError('`kept_columns` must be either a bool, a column name as a string, or '
                            'a list of column names')

        from ._rolling import Roller
        return Roller(self, left, right + 1, min_window, kept_columns)

    def _ipython_key_completions_(self):
        return self._columns.tolist()

    @property
    def str(self):
        return StringClass(self)

    @property
    def dt(self):
        return DateTimeClass(self)

    @property
    def td(self):
        return TimeDeltaClass(self)

    def append(self, objs, axis: str = 'rows', *args, **kwargs):
        """
        Append new rows or columns to the DataFrame.

        Parameters
        ----------
        objs: Dictionary, DataFrame, or list of DataFrames
            Only columns may be appended when a dictionary is used. The keys
            must be the strings of new column names and the values must either be a
            callable, a scalar, an array, or DataFrame. If the value of the dictionary
            is a callable, the *args, and **kwargs are passed to it. It must return a scalar,
            an array of a DataFrame

            If a list of DataFrames is passed

        axis: 'rows' or 'columns'

        args: passed to
        kwargs

        Returns
        -------

        """
        axis_int: int = utils.convert_axis_string(axis)

        if isinstance(objs, dict):
            if axis_int == 0:
                raise NotImplementedError('Using a dictionary of strings mapped to functions '
                                          'is only available for adding columns')
            for k, v in objs.items():
                if not isinstance(k, str):
                    raise TypeError('The keys of the `objs` dict must be a string')

            n = self.shape[0]
            appended_data = {}
            df_new = self.copy()
            for col_name, func in objs.items():
                if isinstance(func, Callable):
                    result = func(self, *args, **kwargs)
                else:
                    result = func

                dtype = _va.get_kind(result)
                if dtype == '':
                    if isinstance(result, DataFrame):
                        if result.size == 1:
                            dtype = list(result._data.keys())[0]
                            arr = np.repeat(result[0, 0], n)
                        elif result.shape[1] != 1:
                            raise ValueError('Returned DataFrame from function mapped from '
                                             f'{col_name} did not return a single column DataFrame')
                        else:
                            dtype = list(result._data.keys())[0]
                            arr = result._data[dtype]

                    elif isinstance(result, ndarray):
                        if result.size == 1:
                            dtype = result.dtype.kind
                            np.repeat(result.flat[0], n)
                        elif (result.ndim == 2 and result.shape[1] != 1) or result.ndim > 2:
                            raise ValueError('Your returned array must have only one column')
                        else:
                            dtype = result.dtype.kind
                            arr = result
                    else:
                        raise TypeError("The return type from the function mapped from "
                                        f"column {col_name} was not a scalar, DataFrame, or array")
                else:
                    arr = np.repeat(result, n)

                if len(arr) < n:
                    arr_old = arr
                    if dtype in 'ifb':
                        arr = np.full((n, 1), nan, dtype='float64')
                    elif dtype == 'M':
                        arr = np.full((n, 1), NaT, dtype='datetime64[ns]')
                    elif dtype == 'm':
                        arr = np.full((n, 1), NaT, dtype='timedelta64[ns]')
                    elif dtype == 'S':
                        arr = np.empty((n, 1), dtype='O')

                    if arr_old.ndim == 1:
                        arr_old = arr_old[:, np.newaxis]
                    arr[:len(arr_old)] = arr_old
                elif len(arr) > n:
                    arr = arr[:n]

                appended_data[col_name] = (dtype, arr)

            new_cols = []
            new_column_info = df_new._column_info
            new_data = df_new._data
            extra_cols = 0
            for col_name, (dt, arr) in appended_data.items():
                if col_name in new_column_info:
                    old_dtype, loc, order = self._get_col_dtype_loc_order(col_name)  # type: str, int, int
                    if old_dtype == dt:
                        new_data[old_dtype][:, loc] = arr
                    else:
                        new_data[old_dtype] = np.delete(new_data[old_dtype], loc, 1)
                        new_loc = new_data[dt].shape[1]
                        new_data[dt] = np.column_stack((new_data[dt], arr))
                        new_column_info[col_name] = utils.Column(dt, new_loc, order)
                else:
                    loc = new_data[dt].shape[1]
                    new_column_info[col_name] = utils.Column(dt, loc, self.shape[1] + extra_cols)
                    extra_cols += 1
                    new_data[dt] = np.column_stack((new_data[dt], arr))
                    new_cols.append(col_name)

            new_columns = np.append(self._columns, new_cols)

        elif isinstance(objs, (DataFrame, list)):
            if isinstance(objs, DataFrame):
                objs = [self, objs]
            else:
                objs = [self] + objs
            for obj in objs:
                if not isinstance(obj, DataFrame):
                    raise TypeError('`Each item in the `objs` list must be a DataFrame`')

            def get_final_dtype(col_dtype):
                if len(col_dtype) == 1 or (len(
                        col_dtype) == 2 and None in col_dtype and 'i' not in col_dtype and 'b'
                                           not in col_dtype):
                    return next(iter(col_dtype - {None}))
                else:
                    if 'm' in col_dtype:
                        raise TypeError(f'The DataFrames that you are appending togethere have '
                                        f'a timedelta64[ns] column and another type in column '
                                        f'number {i}. When appending timedelta64[ns], all '
                                        f'columns must have that type.')
                    elif 'M' in col_dtype:
                        raise TypeError(f'The DataFrames that you are appending togethere have '
                                        f'a datetime64[ns] column and another type in column '
                                        f'number {i}. When appending datetime64[ns], all '
                                        f'columns must have that type.')
                    elif 'S' in col_dtype:
                        raise TypeError('You are trying to append a string column with a non-'
                                        'string column. Both columns must be strings.')
                    elif 'f' in col_dtype or None in col_dtype:
                        return 'f'
                    elif 'i' in col_dtype:
                        return 'i'
                    else:
                        raise ValueError('This error should never happen.')

            if axis_int == 0:
                # append new rows
                ncs = []
                nrs = []
                total = 0
                for obj in objs:
                    ncs.append(obj.shape[1])
                    total += len(obj)
                    nrs.append(total)

                nc = max(ncs)
                nr = nrs[-1]
                new_column_info = {}
                new_columns = []
                data_pieces = defaultdict(list)
                loc_count = defaultdict(int)
                final_col_dtypes = []

                for i in range(nc):
                    col_dtype = set()
                    piece = []
                    has_appended_column = False
                    for ncol, obj in zip(ncs, objs):
                        if i < ncol:
                            col = obj._columns[i]
                            if not has_appended_column:
                                has_appended_column = True
                                new_columns.append(col)
                            dtype, loc = obj._get_col_dtype_loc(col)  # type: str, int
                            col_dtype.add(dtype)
                            piece.append(obj._data[dtype][:, loc])
                        else:
                            piece.append(None)
                            col_dtype.add(None)
                    dtype = get_final_dtype(col_dtype)
                    final_col_dtypes.append(dtype)
                    data_pieces[dtype].append(piece)
                    loc = loc_count[dtype]
                    new_column_info[new_columns[-1]] = utils.Column(dtype, loc, i)
                    loc_count[dtype] += 1

                new_data = {}
                for dtype, pieces in data_pieces.items():
                    make_fast_empty = True
                    for piece in pieces:
                        for p in piece:
                            if p is None:
                                make_fast_empty = False
                                break
                    ct = len(pieces)
                    if dtype in 'bi':
                        dtype_word = utils.convert_kind_to_numpy(dtype)
                        new_data[dtype] = np.empty((nr, ct), dtype=dtype_word, order='F')
                    elif dtype == 'f':
                        if make_fast_empty:
                            new_data[dtype] = np.empty((nr, ct), dtype='float64', order='F')
                        else:
                            new_data[dtype] = np.full((nr, ct), nan, dtype='float64', order='F')
                    elif dtype == 'S':
                        new_data[dtype] = np.empty((nr, ct), dtype='O', order='F')
                    elif dtype == 'm':
                        if make_fast_empty:
                            new_data[dtype] = np.empty((nr, ct), dtype='timedelta64[ns]', order='F')
                        else:
                            new_data[dtype] = np.full((nr, ct), NaT, dtype='timedelta64[ns]',
                                                      order='F')
                    elif dtype == 'M':
                        if make_fast_empty:
                            new_data[dtype] = np.empty((nr, ct), dtype='datetime64[ns]', order='F')
                        else:
                            new_data[dtype] = np.full((nr, ct), NaT, dtype='datetime64[ns]',
                                                      order='F')
                    # pieces is a list of lists
                    for loc, piece in enumerate(pieces):
                        # each p is a one dimensional slice of an array
                        for i, p in enumerate(piece):
                            if p is None:
                                continue
                            if i == 0:
                                left = 0
                            else:
                                left = nrs[i - 1]
                            right = nrs[i]
                            new_data[dtype][left:right, loc] = p

                new_columns = np.array(new_columns, dtype='O')
            else:
                # append new columns
                ncs = []
                nrs = []
                total = 0
                col_maps = []
                new_columns = []
                col_set = set()
                for obj in objs:
                    total += obj.shape[1]
                    ncs.append(total)
                    nrs.append(len(obj))
                    col_map = {}
                    for col in obj._columns:
                        new_col = col
                        i = 1
                        while new_col in col_set:
                            new_col = col + '_' + str(i)
                            i += 1
                        col_map[col] = new_col
                        col_set.add(new_col)
                        new_columns.append(new_col)
                    col_maps.append(col_map)

                nc = nrs[-1]
                nr = max(nrs)
                new_column_info = {}
                data_dict = defaultdict(list)
                loc_count = defaultdict(int)
                new_column_info = {}
                new_data = {}

                i = 0
                for nrow, obj, col_map in zip(nrs, objs, col_maps):
                    for col, dtype, loc, col_arr in obj._col_info_iter(with_arr=True):  # type: str, str, int, int
                        if nrow < nr and dtype in 'bi':
                            dtype = 'f'
                        loc = len(data_dict[dtype])
                        data_dict[dtype].append(col_arr)
                        new_column_info[col_map[col]] = utils.Column(dtype, loc, i)
                        i += 1

                for dtype, data in data_dict.items():
                    if dtype == 'b':
                        new_data[dtype] = np.empty((nr, len(data)), dtype='bool', order='F')
                    elif dtype == 'i':
                        new_data[dtype] = np.empty((nr, len(data)), dtype='int64', order='F')
                    elif dtype == 'f':
                        new_data[dtype] = np.full((nr, len(data)), nan, dtype='float64', order='F')
                    elif dtype == 'S':
                        new_data[dtype] = np.empty((nr, len(data)), dtype='O', order='F')
                    elif dtype == 'm':
                        new_data[dtype] = np.full((nr, len(data)), NaT, dtype='timedelta64[ns]',
                                                  order='F')
                    elif dtype == 'M':
                        new_data[dtype] = np.full((nr, len(data)), NaT, dtype='datetime64[ns]',
                                                  order='F')

                    for i, arr in enumerate(data):
                        new_data[dtype][:len(arr), i] = arr
                new_columns = np.array(new_columns, dtype='O')
        else:
            raise TypeError('`objs` must be either a dictionary, '
                            'a DataFrame or a list of DataFrames. '
                            f'You passed in a {type(objs).__name__}')

        return self._construct_from_new(new_data, new_column_info, new_columns)

    def replace(self, replace_dict):
        if not isinstance(replace_dict, dict):
            raise TypeError('`replace_dict` must be a dictionary')

        # boolean, number, str, datetime, timedelta, missing
        is_all = False
        is_one = False

        specific_replacement = defaultdict(list)
        dtype_conversion = {}

        def get_replacement(key, val):
            if isinstance(key, np.timedelta64):
                if not isinstance(val, np.timedelta64):
                    raise TypeError(f'Cannot replace a timedelta64 with {type(val)}')
                dtype = 'm'
                dtype_conversion['m'] = 'm'
            elif isinstance(key, (bool, np.bool_)):
                if isinstance(val, (bool, np.bool_)):
                    if dtype_conversion.get('b', None) != 'f':
                        dtype_conversion['b'] = 'b'
                elif isinstance(val, (int, np.integer)):
                    if dtype_conversion.get('b', None) != 'f':
                        dtype_conversion['b'] = 'i'
                elif isinstance(val, (float, np.floating)):
                    dtype_conversion['b'] = 'f'
                else:
                    raise TypeError(f'Cannot replace a boolean with {type(val)}')
                dtype = 'b'

            elif isinstance(key, (int, np.integer)):
                if isinstance(val, (bool, np.bool_, int, np.integer)):
                    dtype_conversion['i'] = 'i'
                elif isinstance(val, (float, np.floating)):
                    dtype_conversion['i'] = 'f'
                else:
                    raise TypeError(f'Cannot replace an integer with {type(val)}')
                dtype = 'i'
                # need to do replacement for floats as well as ints
                dtype_conversion['f'] = 'f'

            elif isinstance(key, (float, np.floating)):
                if not isinstance(val, (bool, np.bool_, int, np.integer, float, np.floating)):
                    raise TypeError(f'Cannot replace a float with {type(val)}')
                dtype = 'f'
                dtype_conversion['f'] = 'f'
            elif isinstance(key, str) or key is None:
                if not isinstance(val, str) and val is not None:
                    raise TypeError(f'Cannot replace a str with {type(val)}')
                dtype = 'S'
                dtype_conversion['S'] = 'S'
            elif isinstance(key, np.datetime64):
                if not isinstance(val, np.datetime64):
                    raise TypeError(f'Cannot replace a datetime64 with {type(val)}')
                dtype = 'M'
                dtype_conversion['M'] = 'M'
            else:
                raise TypeError(f'Unknown replacement type: {type(key)}')

            specific_replacement[dtype].append((key, val))
            if dtype == 'i':
                specific_replacement['f'].append((key, val))

        def get_replacement_col(col_dtype, to_repl, replace_val, col):
            if col_dtype == 'm':
                if not isinstance(replace_val, np.timedelta64):
                    raise TypeError(f'Cannot replace a timedelta64 with {type(replace_val)}')
                dtype_conversion[col] = 'm'
            elif col_dtype == 'b':
                if isinstance(replace_val, (bool, np.bool_)):
                    if dtype_conversion.get('b', None) not in 'if':
                        dtype_conversion[col] = 'b'
                elif isinstance(replace_val, (int, np.integer)):
                    if dtype_conversion.get('b', None) != 'f':
                        dtype_conversion[col] = 'i'
                elif isinstance(replace_val, (float, np.floating)):
                    dtype_conversion[col] = 'f'
                else:
                    raise TypeError(f'Cannot replace a boolean with {type(replace_val)}')
            elif col_dtype == 'i':
                if isinstance(replace_val, (bool, np.bool_, int, np.integer)):
                    dtype_conversion[col] = 'i'
                elif isinstance(replace_val, (float, np.floating)):
                    dtype_conversion[col] = 'f'
                else:
                    raise TypeError(f'Cannot replace an integer with {type(replace_val)}')
            elif col_dtype == 'f':
                if not isinstance(replace_val,
                                  (bool, np.bool_, int, np.integer, float, np.floating)):
                    raise TypeError(f'Cannot replace a float with {type(replace_val)}')
                dtype_conversion[col] = 'f'
            elif col_dtype == 'S':
                if not isinstance(replace_val, str) and replace_val is not None:
                    raise TypeError(f'Cannot replace a str with {type(replace_val)}')
                dtype_conversion[col] = 'S'
            elif col_dtype == 'M':
                if not isinstance(replace_val, np.datetime64):
                    raise TypeError(f'Cannot replace a datetime64 with {type(replace_val)}')
                dtype_conversion[col] = 'M'

            specific_replacement[col].append((to_repl, replace_val))

        def check_to_replace_type(key, col, col_dtype):
            keys = key
            if not isinstance(keys, tuple):
                keys = (keys,)

            for key in keys:
                if col_dtype == 'b' and not isinstance(key, (bool, np.bool_)):
                    raise ValueError(f'Column "{col}" is boolean and you are trying to '
                                     f'replace {key}, which is of type {type(key)}')
                elif col_dtype == 'i' and not isinstance(key, (bool, np.bool_, int, np.integer)):
                    raise ValueError(f'Column "{col}" is an int and you are trying to '
                                     f'replace {key}, which is of type {type(key)}')
                elif col_dtype == 'f' and not isinstance(key, (bool, np.bool_, int, np.integer,
                                                               float, np.floating)):
                    raise ValueError(f'Column "{col}" is a float and you are trying to '
                                     f'replace {key}, which is of type {type(key)}')
                elif col_dtype == 'S' and not isinstance(key, str):
                    raise ValueError(f'Column "{col}" is a str and you are trying to '
                                     f'replace {key}, which is of type {type(key)}')
                elif col_dtype == 'M' and not isinstance(key, np.datetime64):
                    raise ValueError(f'Column "{col}" is a datetime64 and you are trying to '
                                     f'replace {key}, which is of type {type(key)}')
                elif col_dtype == 'm' and not isinstance(key, np.timedelta64):
                    raise ValueError(f'Column "{col}" is a timedelta64 and you are trying to '
                                     f'replace {key}, which is of type {type(key)}')

        col_set = set(self._columns)
        for key, val in replace_dict.items():
            if isinstance(val, dict):
                if is_all:
                    raise TypeError('`replace_dict` must either be a dictionary of dictionaries '
                                    'or a dictionary of scalars (or tuples) mapped to replacement '
                                    'values. You cannot mix the two.')
                is_one = True
            else:
                if is_one:
                    raise TypeError('`replace_dict` must either be a dictionary of dictionaries '
                                    'or a dictionary of scalars (or tuples) mapped to replacement '
                                    'values. You cannot mix the two.')
                is_all = True

            if is_all:
                keys = key
                if not isinstance(keys, tuple):
                    keys = (keys,)

                for key in keys:
                    get_replacement(key, val)
            else:
                # this is a column by column replacement with a dict of dicts
                # key is the column name here
                if key not in col_set:
                    raise ValueError(f'Column "{key}" not found in DataFrame')
                col_dtype = self._column_info[key].dtype
                for k, v in val.items():
                    check_to_replace_type(k, key, col_dtype)
                    get_replacement_col(col_dtype, k, v, key)

        if is_all:
            used_dtypes = set()
            data_dict = defaultdict(list)
            new_column_info = {}
            cols, locs, ords = self._get_all_dtype_info()

            used_dtypes = set(dtype_conversion)
            converted_dtypes = set(dtype_conversion.values())
            cur_dtype_loc = defaultdict(int)

            for dtype, data in self._data.items():
                if dtype not in used_dtypes:
                    if dtype in converted_dtypes:
                        data_dict[dtype].append(data)
                    else:
                        data_dict[dtype].append(data.copy('F'))

                    for col, loc, order in zip(cols[dtype], locs[dtype], ords[dtype]):
                        new_column_info[col] = utils.Column(dtype, loc, order)
                    cur_dtype_loc[dtype] = data.shape[1]

            for dtype, new_dtype in dtype_conversion.items():
                used_dtypes.add(dtype)
                old_name = utils.convert_dtype_to_func_name(dtype)
                new_name = utils.convert_dtype_to_func_name(new_dtype)
                func_name = f'replace_{old_name}_with_{new_name}'

                dtype_word = utils.convert_kind_to_numpy(dtype)
                new_dtype_word = utils.convert_kind_to_numpy(new_dtype)
                if new_dtype_word == 'float':
                    dtype_word = 'float'
                elif new_dtype_word == 'int' and dtype_word == 'bool':
                    dtype_word = 'int'
                cur_replacement = specific_replacement[dtype]
                n = len(cur_replacement)
                to_replace = np.empty(n, dtype=dtype_word)
                replacements = np.empty(n, dtype=new_dtype_word)

                for i, (repl, repl_with) in enumerate(cur_replacement):
                    to_replace[i] = repl
                    replacements[i] = repl_with
                func = getattr(_repl, func_name)
                data = self._data[dtype]
                if dtype in 'mM':
                    data_dict[new_dtype].append(func(data.view('int64'), to_replace.view('int64'),
                                                     replacements.view('int64')))
                else:
                    data_dict[new_dtype].append(func(data, to_replace, replacements))
                loc_add = cur_dtype_loc[new_dtype]
                for col, loc, order in zip(cols[dtype], locs[dtype], ords[dtype]):
                    new_column_info[col] = utils.Column(new_dtype, loc + loc_add, order)
                cur_dtype_loc[new_dtype] += data.shape[1]

            new_data = {}
            for dtype, data in data_dict.items():
                new_data[dtype] = np.column_stack(data)

            new_columns = self._columns.copy()
        else:
            unused_dtypes = set(self._data)
            used_dtypes = set()
            for col, dtype in dtype_conversion.items():
                old_dtype = self._column_info[col].dtype
                unused_dtypes.discard(dtype)
                unused_dtypes.discard(old_dtype)
                used_dtypes.add(dtype)
                used_dtypes.add(old_dtype)

            cols, locs, ords = self._get_all_dtype_info()
            new_columns = self._columns.copy()
            new_column_info = {}
            new_data = {}

            for dtype in unused_dtypes:
                new_data[dtype] = self._data[dtype].copy('F')
                for col, loc, order in zip(cols[dtype], locs[dtype], ords[dtype]):
                    new_column_info[col] = utils.Column(dtype, loc, order)

            used_dtype_ncols = {}
            for dtype in used_dtypes:
                used_dtype_ncols[dtype] = self._data[dtype].shape[1]

            for col, new_dtype in dtype_conversion.items():
                used_dtype_ncols[new_dtype] += 1
                old_dtype = self._column_info[col].dtype
                used_dtype_ncols[old_dtype] -= 1

            for dtype, ncol in used_dtype_ncols.items():
                nr = len(self)
                dtype_word = utils.convert_kind_to_numpy(dtype)
                new_data[dtype] = np.empty((nr, ncol), dtype=dtype_word)

            cur_dtype_loc = defaultdict(int)
            for dtype in used_dtypes:
                for col, old_loc, order in zip(cols[dtype], locs[dtype], ords[dtype]):
                    if col not in specific_replacement:
                        loc = cur_dtype_loc[dtype]
                        new_column_info[col] = utils.Column(dtype, loc, order)
                        new_data[dtype][:, loc] = self._data[dtype][:, old_loc]
                        cur_dtype_loc[dtype] += 1
                    else:
                        old_dtype, old_loc, order = self._column_info[col].values
                        new_dtype = dtype_conversion[col]
                        old_name = utils.convert_dtype_to_func_name(old_dtype)
                        new_name = utils.convert_dtype_to_func_name(new_dtype)
                        func_name = f'replace_{old_name}_with_{new_name}'

                        cur_replacement = specific_replacement[col]
                        n = len(cur_replacement)
                        dtype_word = utils.convert_kind_to_numpy(old_dtype)
                        new_dtype_word = utils.convert_kind_to_numpy(new_dtype)
                        if new_dtype_word == 'float':
                            dtype_word = 'float'
                        elif new_dtype_word == 'int' and dtype_word == 'bool':
                            dtype_word = 'int'
                        to_replace = np.empty(n, dtype=dtype_word)
                        replacements = np.empty(n, dtype=new_dtype_word)

                        for i, (repl, repl_with) in enumerate(cur_replacement):
                            to_replace[i] = repl
                            replacements[i] = repl_with
                        func = getattr(_repl, func_name)
                        data = self._data[old_dtype][:, old_loc][:, np.newaxis]
                        loc = cur_dtype_loc[new_dtype]
                        if old_dtype in 'mM':
                            data = data.view('int64')
                            to_replace = to_replace.view('int64'),
                            replacements = replacements.view('int64')
                        new_data[new_dtype][:, loc] = func(data, to_replace, replacements).squeeze()
                        new_column_info[col] = utils.Column(new_dtype, loc, order)
                        cur_dtype_loc[new_dtype] += 1

        return self._construct_from_new(new_data, new_column_info, new_columns)

    def pivot(self, row, column, value=None, aggfunc=None, normalize=None):
        self._validate_column_name(row)
        self._validate_column_name(column)
        if value is None:
            if aggfunc is not None:
                if aggfunc != 'size':
                    raise ValueError('You provided a value for `aggfunc` but did not provide a '
                                     '`value` column to aggregate.')
            else:
                raise ValueError('`value` and `aggfunc` cannot both be `None`')
        else:
            self._validate_column_name(value)
        group = [row, column]
        temp_col_name = '____TEMP____'
        col_set = set(group)
        i = 0
        while temp_col_name in col_set:
            temp_col_name = '____TEMP____' + str(i)
            i += 1
        if aggfunc is None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df_group = self.groupby(group).size()
            size_col = df_group._columns[-1]
            dtype, loc = df_group._get_col_dtype_loc(size_col)  # type: str, int
            max_size = df_group._data[dtype][:, loc].max()
            if max_size > 1:
                raise ValueError('You did not provide an `aggfunc` which means that each '
                                 f'combination of {row} and {column} must have at most one '
                                 'value')
            df_group = self[:, [row, column, value]]
            temp_col_name = value
        elif aggfunc == 'size':
            df_group = self.groupby(group).size()
            temp_col_name = df_group.columns[-1]
        else:
            df_group = self.groupby(group).agg((aggfunc, value, temp_col_name))

        row_idx, row_names = df_group.factorize(row)
        col_idx, col_names = df_group.factorize(column)

        dtype, loc = df_group._get_col_dtype_loc(temp_col_name)  # type: str, int
        value_arr = df_group._data[dtype][:, loc]

        new_data = {}
        if dtype in 'ib':
            if aggfunc != 'size' and len(row_idx) != len(row_names) * len(col_names):
                value_arr = value_arr.astype('float64')
                dtype = 'f'

        func_name = 'pivot_' + utils.convert_dtype_to_func_name(dtype)
        row_name_dtype = row_names.dtype.kind
        pivot_result = getattr(_pivot, func_name)(row_idx, len(row_names),
                                                  col_idx, len(col_names), value_arr)
        pivot_result_dtype = pivot_result.dtype.kind
        loc_add = 0
        if row_name_dtype == pivot_result_dtype:
            new_data[dtype] = np.column_stack((row_names, pivot_result))
            loc_add += 1
        else:
            new_data[row_name_dtype] = row_names[:, np.newaxis]
            new_data[pivot_result_dtype] = pivot_result

        if col_names.dtype.kind != 'S':
            col_names = col_names.astype('U').astype('O')

        # avoid row names collision
        new_column_info = {row: utils.Column(row_name_dtype, 0, 0)}

        for i, col in enumerate(col_names):
            new_column_info[col] = utils.Column(dtype, i + loc_add, i + 1)

        new_columns = np.concatenate(([row], col_names))

        # TODO: Optimize this
        df_unsorted = self._construct_from_new(new_data, new_column_info, new_columns)
        df_unsorted = df_unsorted.sort_values(row)
        sorted_cols = np.concatenate(([new_columns[0]], np.sort(new_columns[1:])))
        return df_unsorted[:, sorted_cols]

    def melt(self, id_vars=None, value_vars=None, var_name='variable', value_name='value'):

        def check_string_or_list(vals, name, none_possible=True):
            if isinstance(vals, str):
                return [vals]
            elif isinstance(vals, list):
                return vals
            elif isinstance(vals, np.ndarray):
                return vals.tolist()
            elif vals is None and none_possible:
                return []
            else:
                raise TypeError('`{name}` must be a string or list of strings')

        # much easier to handle parameters when they are all lists
        id_vars = check_string_or_list(id_vars, 'id_vars')
        value_vars = check_string_or_list(value_vars, 'value_vars')
        var_name = check_string_or_list(var_name, 'var_name', False)
        value_name = check_string_or_list(value_name, 'value_name', False)

        if id_vars:
            self._validate_column_name_list(id_vars)
        id_vars_set = set(id_vars)

        if value_vars == []:
            value_vars = [col for col in self._columns if col not in id_vars_set]

        if not isinstance(value_vars[0], list):
            value_vars = [value_vars]

        n_groups = len(value_vars)

        # When a list of lists is passed, assume multiple melt groups
        if n_groups != len(var_name):
            if len(var_name) != 1:
                raise ValueError('Number of inner lists of value_vars must '
                                 'equal length of var_name '
                                 f'{len(value_vars)} != {len(var_name)}')
            else:
                var_name = [var_name[0] + '_' + str(i) for i in range(n_groups)]

        for vn in var_name:
            if not isinstance(vn, str):
                raise TypeError('`var_name` must be a string or list of strings')

        if n_groups != len(value_name):
            if len(value_name) != 1:
                raise ValueError('Number of inner lists of value_vars must '
                                 'equal length of value_name '
                                 f'{n_groups} != {len(value_name)}')
            else:
                value_name = [value_name[0] + '_' + str(i) for i in range(n_groups)]

        for vv in value_name:
            if not isinstance(vv, str):
                raise TypeError('`value_name` must be a string or list of strings')

        # get the total number of columns in each melt group
        # This is not just the length of the list because this function
        # handles columns with the same names, which is common when
        # using multiindex frames
        value_vars_length = [len(vv) for vv in value_vars]

        # Need the max number of columns for all the melt groups to
        # correctly append NaNs to end of unbalanced melt groups
        max_group_len = max(value_vars_length)
        data_dict = defaultdict(list)
        new_column_info = {}

        # validate all columns in value_vars and ensure they don't appear as id_vars
        col_set = set(self._columns)

        if len(id_vars_set) != len(id_vars):
            raise ValueError('`id_vars` cannot contain duplicate column names')
        all_value_vars_set = set()
        for vv in value_vars:
            for col in vv:
                if col in id_vars_set:
                    raise ValueError(f'Column "{col}" cannot be both an id_var and a value_var')
                if col in all_value_vars_set:
                    raise ValueError(f'column "{col}" is already a value_var')
                self._validate_column_name(col)
                all_value_vars_set.add(col)

        cur_order = 0
        new_columns = []
        # get the id_vars (non-melted) columns
        for i, col in enumerate(id_vars):
            dtype, loc = self._get_col_dtype_loc(col)  # type: str, int
            arr = self._data[dtype][:, loc]
            arr = np.tile(arr, max_group_len)
            new_loc = len(data_dict[dtype])
            data_dict[dtype].append(arr)
            new_column_info[col] = utils.Column(dtype, new_loc, i)
            cur_order += 1
            new_columns.append(col)

        # individually melt each group
        vars_zipped = zip(value_vars, var_name, value_name, value_vars_length)
        fill_empty_arr = {}
        for i, (val_v, var_n, val_n, vvl) in enumerate(vars_zipped):
            dtype_loc = defaultdict(list)
            for col in val_v:
                dtype, loc = self._get_col_dtype_loc(col)  # type: str, int
                dtype_loc[dtype].append(loc)

            if len(dtype_loc) > 1:
                dt_string = ''
                if 'S' in dtype_loc:
                    dt_string = 'string'
                elif 'm' in dtype_loc:
                    dt_string = 'timedelta'
                elif 'M' in dtype_loc:
                    dt_string = 'datetime'
                elif 'f' in dtype_loc:
                    new_dtype = 'f'
                elif 'i' in dtype_loc:
                    new_dtype = 'i'
                elif 'b' in dtype_loc:
                    new_dtype = 'b'

                if dt_string:
                    raise TypeError(f'You are attempting to melt columns with a mix of {dt_string} '
                                    f'and non-{dt_string} types. You can only melt string columns '
                                    f'if '
                                    'they are all string columns')
            else:
                new_dtype = dtype

            if len(val_v) < max_group_len:
                if dtype in 'ib':
                    new_dtype = 'f'
                fill_empty_arr[new_dtype] = True

            # add the variable column - column names into column values
            cur_loc = len(data_dict['S'])
            variable_vals = np.repeat(np.array(val_v, dtype='O'), len(self))
            data_dict['S'].append(variable_vals)
            new_column_info[var_n] = utils.Column('S', cur_loc, cur_order)
            cur_order += 1
            new_columns.append(var_n)

            if len(dtype_loc) == 1:
                locs = dtype_loc[dtype]
                data = self._data[dtype][:, locs].flatten('F')
            else:
                all_data = []
                for dtype, loc in dtype_loc.items():
                    all_data.append(self._data[dtype][:, loc])
                data = np.concatenate(all_data)

            cur_loc = len(data_dict[new_dtype])
            data_dict[new_dtype].append(data)
            new_column_info[val_n] = utils.Column(new_dtype, cur_loc, cur_order)
            cur_order += 1
            new_columns.append(val_n)

        new_columns = np.array(new_columns, dtype='O')
        N = max_group_len * len(self)
        new_data = {}
        for dtype, data_list in data_dict.items():
            size = (N, len(data_list))
            if fill_empty_arr.get(dtype, False):
                # need to make full array with nans
                if dtype == 'f':
                    arr = np.full(size, nan, dtype='float64', order='F')
                elif dtype == 'S':
                    arr = np.empty(size, dtype='O', order='F')
                elif dtype == 'm':
                    arr = np.full(size, NaT, dtype='timedelta64', order='F')
                elif dtype == 'M':
                    arr = np.full(size, NaT, dtype='datetime64', order='F')
            else:
                dtype_word = utils.convert_kind_to_numpy(dtype)
                arr = np.empty(size, dtype=dtype_word, order='F')

            for i, data in enumerate(data_list):
                arr[:len(data), i] = data
            new_data[dtype] = arr
        return self._construct_from_new(new_data, new_column_info, new_columns)

    def to_csv(self, fp, sep=','):

        if not isinstance(sep, str) or len(sep) != 1:
            raise TypeError('`sep` must be a one-character string')

        def get_dtype_arr():
            arr = np.empty(self.shape[1], dtype='int64')

            for i, (col, col_obj) in enumerate(self._column_info.items()):
                if col_obj.dtype == 'f':
                    arr[i] = 0
                elif col_obj.dtype == 'S':
                    arr[i] = 1
                elif col_obj.dtype in 'mM':
                    arr[i] = 2
                elif col_obj.dtype in 'ib':
                    arr[i] = 3
            return arr

        dtypes_arr = get_dtype_arr()
        values = self.values
        columns = self._columns.astype('O')
        _of.to_csv(values, columns, dtypes_arr, fp, sep)

    def join(self, right, how='inner', on=None, left_on=None, right_on=None):
        # if not isinstance(right, DataFrame):
        #     raise TypeError('`right` must be a DataFrame')

        def check_cols(df, on_cols, name):
            if isinstance(on_cols, str):
                df._validate_column_name(on_cols)
                on_cols = [on_cols]
            elif isinstance(on_cols, list):
                num_cols = len(on_cols)
                self._validate_column_name_list(on_cols)
            else:
                raise TypeError(f'`{name}` must be either a string or a list of strings')
            return on_cols

        if on is None:
            if left_on is None:
                if right_on is None:
                    # default to all columns with same names
                    left_cols = right_cols = self._column_info.keys() & right._column_info.keys()
                    if not left_cols:
                        raise ValueError('You did not provide any column names for joining and '
                                         'there are no common column names')
                else:
                    raise ValueError('You must provide both `left_on` and `right_on` and not '
                                     'just one of them.')
            else:
                if right_on is None:
                    raise ValueError('You must provide both `left_on` and `right_on` and not '
                                     'just one of them.')
                else:
                    left_cols = check_cols(self, left_on, 'left_on')
                    right_cols = check_cols(right, right_on, 'right_on')
                    if len(left_cols) != len(right_cols):
                        raise ValueError('The number of columns in `left_on` and `right_on` are '
                                         'not equal. {len(left_cols)} != {len(right_cols)}')
        else:
            if left_on is not None or right_on is not None:
                raise ValueError('When providing a value for `on`, you cannot provide a value for '
                                 '`left_on` or `right_on` and vice-versa')
            left_cols = check_cols(self, on, 'on')
            right_cols = check_cols(right, on, 'on')

        # they both have same number of columns so it doesn't matter which one we use here
        num_cols = len(left_cols)
        has_multiple_join_cols = num_cols > 1

        if num_cols == 0:
            raise ValueError('You must provide at least one column to join on')

        if num_cols == 1:
            left_col = left_cols[0]
            left_dtype, left_loc, _ = self._column_info[left_col].values

            right_col = right_cols[0]
            right_dtype, right_loc, _ = right._column_info[right_col].values

            if left_dtype != right_dtype:
                if left_dtype in 'OmM':
                    dtype_name = utils.convert_dtype_to_func_name(left_dtype)
                    raise TypeError(f'The calling DataFrame join column is a {dtype_name} while '
                                    'the `right` DataFrame join column is not. They both must be '
                                    f'{dtype_name} or other compatible types')
                elif right_dtype in 'OmM':
                    dtype_name = utils.convert_dtype_to_func_name(right_dtype)
                    raise TypeError(f'The `right` DataFrame join column is a {dtype_name} while '
                                    'the calling DataFrame join column is not. They both must be '
                                    f'{dtype_name} or other compatible types.')

            left_arr = self._data[left_dtype][:, left_loc]
            right_arr = right._data[right_dtype][:, right_loc]

            if left_dtype == 'S':
                func_name = 'join_str_1d'
            elif left_dtype in 'mM':
                func_name = 'join_float_1d'
                is_nat = np.isnat(left_arr)
                left_arr = left_arr.astype('float64')
                left_arr[is_nat] = nan
                right_arr = right_arr.view('float64')
                right_arr[is_nat] = nan
            elif left_dtype == 'f' or right_dtype == 'f':
                func_name = 'join_float_1d'
                if left_dtype != 'f':
                    left_arr = left_arr.astype('float64')
                if right_dtype != 'f':
                    right_arr = right_arr.astype('float64')
            elif left_dtype == 'i' or right_dtype == 'i':
                func_name = 'join_int_1d'
                if left_dtype != 'i':
                    left_arr = left_arr.astype('int64')
                if right_dtype != 'i':
                    right_arr = right_arr.astype('int64')
            else:
                func_name = 'join_bool_1d'
        else:
            # multiple join columns
            left_dtype_locs = defaultdict(list)
            right_dtype_locs = defaultdict(list)
            for left_col, right_col in zip(left_cols, right_cols):
                left_dtype, left_loc, _ = self._column_info[left_col].values
                right_dtype, right_loc, _ = right._column_info[right_col].values

                if left_dtype != right_dtype:
                    if left_dtype in 'OmM':
                        dtype_name = utils.convert_dtype_to_func_name(left_dtype)
                        raise TypeError(
                            f'The calling DataFrame join column is a {dtype_name} while '
                            'the `right` DataFrame join column is not. They both must be '
                            f'{dtype_name} or other compatible types')
                    elif right_dtype in 'OmM':
                        dtype_name = utils.convert_dtype_to_func_name(right_dtype)
                        raise TypeError(
                            f'The `right` DataFrame join column is a {dtype_name} while '
                            'the calling DataFrame join column is not. They both must be '
                            f'{dtype_name} or other compatible types.')

                left_dtype_locs[left_dtype].append(left_loc)
                right_dtype_locs[right_dtype].append(right_loc)

            if 'f' in left_dtype_locs or 'f' in right_dtype_locs:
                final_num_dtype = 'f'
            # for time or date, if its in one its in the other
            elif 'm' in left_dtype_locs or 'M' in left_dtype_locs:
                final_num_dtype = 'f'
            elif 'i' in left_dtype_locs or 'i' in left_dtype_locs:
                final_num_dtype = 'i'
            elif 'b' in left_dtype_locs or 'b' in right_dtype_locs:
                final_num_dtype = 'b'
            else:
                final_num_dtype = ''

            if final_num_dtype:
                ncol_number = 0
                for dtype, locs in left_dtype_locs.items():
                    if dtype == 'S':
                        continue
                    ncol_number += len(locs)
                dtype_word = utils.convert_kind_to_numpy(final_num_dtype)
                left_arr_number = np.empty((len(self), ncol_number), dtype=dtype_word)
                right_arr_number = np.empty((len(right), ncol_number), dtype=dtype_word)

                i_col = 0
                for dtype, locs in left_dtype_locs.items():
                    for loc in locs:
                        left_arr_number[:, i_col] = self._data[dtype][:, loc]
                        i_col += 1

                i_col = 0
                for dtype, locs in right_dtype_locs.items():
                    for loc in locs:
                        right_arr_number[:, i_col] = right._data[dtype][:, loc]
                        i_col += 1

            has_str_join_columns = 'S' in left_dtype_locs

            if has_str_join_columns:
                if final_num_dtype:
                    pass
                else:
                    # only has string columns
                    func_name = 'join_str_2d'

        subtract_common_cols = left_cols == right_cols
        left_col_ct = defaultdict(int)
        right_col_ct = defaultdict(int)
        for col in left_cols:
            dtype = self._column_info[col].dtype
            left_col_ct[dtype] += 1

        for col in right_cols:
            dtype = right._column_info[col].dtype
            right_col_ct[dtype] += 1

        # get number of columns for each dtype
        new_column_info = self._copy_column_info()
        new_column_list = self.columns
        dtype_col_nums = defaultdict(int)
        dtype_cur_loc = defaultdict(int)

        for dtype, data in self._data.items():
            dtype_col_nums[dtype] += data.shape[1]
            dtype_cur_loc[dtype] += data.shape[1]

        for dtype, data in right._data.items():
            dtype_col_nums[dtype] += data.shape[1]
            if subtract_common_cols:
                dtype_col_nums[dtype] -= right_col_ct[dtype]

        cur_order = len(self._columns)
        dtype_right_locs = defaultdict(list)
        for col in right._columns:
            if subtract_common_cols and col in right_cols:
                continue

            dtype, loc = right._get_col_dtype_loc(col)  # type: str, int
            cur_loc = dtype_cur_loc[dtype]
            col_y = col
            if col in self._column_info:
                col_y = col + '_y'
                col_x = col + '_x'
                left_col_idx = new_column_list.index(col)
                new_column_list[left_col_idx] = col_x
                new_column_info[col_x] = new_column_info[col]
                del new_column_info[col]

            new_column_info[col_y] = utils.Column(dtype, cur_loc, cur_order)
            dtype_cur_loc[dtype] += 1
            cur_order += 1
            new_column_list.append(col_y)
            dtype_right_locs[dtype].append(loc)

        if has_multiple_join_cols:
            if final_num_dtype:
                pass
            else:
                # must have str join columns
                left_arr = self._data['S']
                right_arr = right._data['S']
                left_locs = left_dtype_locs['S']
                right_locs = right_dtype_locs['S']
                left_rep, right_idx, n_rows = _join.join_str_2d(left_arr, right_arr, left_locs,
                                                                right_locs)
        else:
            left_rep, right_idx, n_rows = getattr(_join, func_name)(left_arr, right_arr)

        new_data = {}

        for dtype, ncol in dtype_col_nums.items():
            if ncol == 0:
                continue
            if dtype not in right._data:
                data = self._data[dtype]
                arr = np.repeat(data, left_rep, axis=0)
            elif dtype not in self._data:
                data = right._data[dtype]
                locs = dtype_right_locs[dtype]
                arr = data[np.ix_(right_idx, locs)]
            else:
                dtype_word = utils.convert_kind_to_numpy(dtype)
                arr = np.empty((n_rows, ncol), dtype=dtype_word)
                data = np.repeat(self._data[dtype], left_rep, axis=0)
                if data.ndim == 1:
                    data = data[:, np.newaxis]
                arr[:, :data.shape[1]] = data

                locs = dtype_right_locs[dtype]
                if locs:
                    data = right._data[dtype][:, locs]
                    arr[:, data.shape[1]:] = data[right_idx]

            if arr.ndim == 1:
                arr = arr[:, np.newaxis]
            new_data[dtype] = arr

        new_columns = np.array(new_column_list, dtype='O')
        return self._construct_from_new(new_data, new_column_info, new_columns)
