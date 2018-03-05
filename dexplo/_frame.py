from collections import defaultdict, OrderedDict
from math import ceil
import warnings

import numpy as np
from numpy import nan, ndarray
from typing import (Union, Dict, List, Optional, Tuple, Callable, overload,
                    NoReturn, Set, Iterable, Any, TypeVar, Type, Generator)

import dexplo.options as options
import dexplo._utils as utils
from dexplo._libs import (string_funcs as _sf,
                          groupby as _gb,
                          validate_arrays as _va,
                          math as _math,
                          sort_rank as _sr,
                          unique as _uq)
from dexplo import _stat_funcs as stat
from dexplo._strings import StringClass
from dexplo._date import DateTimeClass
from dexplo._date import TimeDeltaClass

DataC = Union[Dict[str, Union[ndarray, List]], ndarray]

# can change to array of strings?
ColumnT = Optional[Union[List[str], ndarray]]
ColInfoT = Dict[str, utils.Column]

IntStr = TypeVar('IntStr', int, str)
IntNone = TypeVar('IntNone', int, None)
ListIntNone = List[IntNone]

ScalarT = TypeVar('ScalarT', int, float, str, bool)

ColSelection = Union[int, str, slice, List[IntStr], 'DataFrame']
RowSelection = Union[int, slice, List[int], 'DataFrame']
Scalar = Union[int, str, bool, float]


class DataFrame(object):
    """
    The DataFrame is the primary data container in dexplo.
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
    >>> df = de.DataFrame({'State': ['TX', 'FL', 'CO'],
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
        self._hasnans: Dict[str, List[ndarray]] = {}

        if isinstance(data, dict):
            self._initialize_columns_from_dict(columns, data)
            self._initialize_data_from_dict(data)

        elif isinstance(data, ndarray):
            num_cols: int = utils.validate_array_type_and_dim(data)
            self._initialize_columns_from_array(columns, num_cols)

            if data.dtype.kind == 'O':
                self._initialize_from_object_array(data)
            else:
                self._initialize_data_from_array(data)

        else:
            raise TypeError('data parameter must be either a dict of arrays or an array')

        self._add_accessors()

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
        new_columns = self._check_column_validity(new_columns)

        len_new: int = len(new_columns)
        len_old: int = len(self._columns)
        if len_new != len_old:
            raise ValueError(f'There are {len_old} columns in the DataFrame. '
                             f'You provided {len_new}')

        new_column_info: ColInfoT = {}
        for old_col, new_col in zip(self._columns, new_columns):
            new_column_info[new_col] = utils.Column(*self._column_info[old_col].values)

        self._column_info: ColInfoT = new_column_info
        self._columns: ColumnT = new_columns

    def _initialize_columns_from_dict(self, columns: ColumnT, data: DataC) -> None:
        """
        Sets the column names when a dictionary is passed to the DataFrame constructor.
        If the columns parameter is not none, its elements must match the dictionary keys.

        Parameters
        ----------
        columns: List or array of strings of column names
        data: Dictionary of lists or 1d arrays

        Returns
        -------
        None

        """
        if columns is None:
            columns = np.array(list(data.keys()))
            columns = self._check_column_validity(columns)
        else:
            columns = self._check_column_validity(columns)
            if set(columns) != set(data.keys()):
                raise ValueError("Column names don't match dictionary keys")

        self._columns: ColumnT = columns

    def _initialize_columns_from_array(self, columns: ColumnT, num_cols: int) -> None:
        """
        When an array or list is passed to the `columns` parameter in the DataFrame constructor

        Parameters
        ----------
        columns : List or array of strings column names
        num_cols : Integer of the number of columns

        Returns
        -------
        None

        """
        if columns is None:
            col_list: List[str] = ['a' + str(i) for i in range(num_cols)]
            self._columns: ColumnT = np.array(col_list, dtype='O')
        else:
            columns = self._check_column_validity(columns)
            if len(columns) != num_cols:
                raise ValueError(f'Number of column names {len(columns)} does not equal '
                                 f'number of columns of data array {num_cols}')
            self._columns: ColumnT = columns

    def _check_column_validity(self, cols: ColumnT) -> ndarray:
        """
        Determine if column names are valid
        Parameters
        ----------
        cols : list or array of strings

        Returns
        -------
        Nothing when valid and raises an error if duplicated or non-string
        """
        if not isinstance(cols, (list, ndarray)):
            raise TypeError('Columns must be a list or an array')
        if isinstance(cols, ndarray):
            cols = utils.try_to_squeeze_array(cols)

        col_set: Set[str] = set()
        for i, col in enumerate(cols):
            if not isinstance(col, str):
                raise TypeError('Column names must be a string')
            if col in col_set:
                raise ValueError(f'Column name {col} is duplicated. Column names must be unique')
            col_set.add(col)
        return np.asarray(cols, dtype='O')

    def _initialize_data_from_dict(self, data: DataC) -> None:
        """
        Sets the _data attribute whenever a dictionary is passed to the `data` parameter in the
        DataFrame constructor. Also sets `_column_info`

        Parameters
        ----------
        data: Dictionary of lists or 1d arrays

        Returns
        -------
        None
        """
        data_dict: Dict[str, List[ndarray]] = defaultdict(list)
        for i, (col, values) in enumerate(data.items()):
            if isinstance(values, list):
                arr: ndarray = utils.convert_list_to_single_arr(values)
            elif isinstance(values, ndarray):
                arr = values
            else:
                raise TypeError('Values of dictionary must be an array or a list')
            arr = utils.maybe_convert_1d_array(arr, col)
            kind: str = arr.dtype.kind
            loc: int = len(data_dict.get(kind, []))
            data_dict[kind].append(arr)
            self._column_info[col] = utils.Column(kind, loc, i)

            if i == 0:
                first_len: int = len(arr)
            elif len(arr) != first_len:
                raise ValueError('All columns must be the same length')

        self._data = self._concat_arrays(data_dict)

    def _initialize_data_from_array(self, data: ndarray) -> None:
        """
        Stores entire array, `data` into `self._data` as one kind

        Parameters
        ----------
        data : A homogeneous array

        Returns
        -------
        None
        """
        kind: str = data.dtype.kind
        if kind == 'U':
            data = data.astype('O')
        elif kind == 'M':
            data = data.astype('datetime64[ns]')
        elif kind == 'm':
            data = data.astype('timedelta64[ns]')

        if data.ndim == 1:
            data = data[:, np.newaxis]

        kind = data.dtype.kind

        # Force array to be fortran ordered
        self._data[kind] = np.asfortranarray(data)
        self._column_info = {col: utils.Column(kind, i, i)
                             for i, col in enumerate(self._columns)}

    def _initialize_from_object_array(self, data: ndarray) -> None:
        """
        Special initialization when array if of kind 'O'. Must check each column individually

        Parameters
        ----------
        data : A numpy object array

        Returns
        -------
        None
        """
        if data.ndim == 1:
            data = data[:, np.newaxis]

        data_dict: Dict[str, List[ndarray]] = defaultdict(list)
        for i, col in enumerate(self._columns):
            arr: ndarray = _va.maybe_convert_object_array(data[:, i], col)
            kind: str = arr.dtype.kind
            loc: int = len(data_dict[kind])
            data_dict[kind].append(arr)
            self._column_info[col] = utils.Column(kind, loc, i)

        self._data = self._concat_arrays(data_dict)

    def _add_accessors(self):
        self.str = StringClass(self)
        self.dt = DateTimeClass(self)
        self.td = TimeDeltaClass(self)

    def _concat_arrays(self, data_dict: Dict[str, List[ndarray]]) -> Dict[str, ndarray]:
        """
        Concatenates the lists for each kind into a single array
        """
        new_data: Dict[str, ndarray] = {}
        for dtype, arrs in data_dict.items():
            if arrs:
                if len(arrs) == 1:
                    new_data[dtype] = arrs[0][:, np.newaxis]
                else:
                    arrs = np.column_stack(arrs)
                    new_data[dtype] = np.asfortranarray(arrs)
        return new_data

    @property
    def values(self) -> ndarray:
        """
        Retrieve a single 2-d array of all the data in the correct column order
        """
        if len(self._data) == 1:
            kind: str = list(self._data.keys())[0]
            order = [self._column_info[col].loc for col in self._columns]
            return self._data[kind][:, order]

        if 'b' in self._data or 'O' in self._data or 'm' in self._data or 'M' in self._data:
            arr_dtype: str = 'O'
        else:
            arr_dtype = 'float64'

        v: ndarray = np.empty(self.shape, dtype=arr_dtype, order='F')

        for col, col_obj in self._column_info.items():
            dtype, loc, order2 = col_obj.values  # type: str, int, int
            col_arr = self._data[dtype][:, loc]
            if dtype == 'M':
                unit = col_arr.dtype.name.replace(']', '').split('[')[1]
                # changes array in place
                _va.make_object_datetime_array(v, col_arr.view('uint64'), order2, unit)
            elif dtype == 'm':
                unit = col_arr.dtype.name.replace(']', '').split('[')[1]
                _va.make_object_timedelta_array(v, col_arr.view('uint64'), order2, unit)
            else:
                v[:, order2] = col_arr
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
        i: int = 0
        for col in self._columns:
            dtype, loc, order = self._column_info[col].values  # type: str, int, int
            if dtype in 'ifb':
                v[:, i] = self._data[dtype][:, loc]
                i += 1
        return v

    def _values_number_drop(self, columns: List[str], dtype_loc: List[Tuple[str, int]],
                            np_dtype: str) -> ndarray:
        """
        Retrieve the array that consists only of integer and floats
        Cov and Corr use this
        """
        shape: Tuple[int, int] = (len(self), len(columns))

        v: ndarray = np.empty(shape, dtype=np_dtype, order='F')
        i: int = 0
        for col, (dtype, loc) in zip(columns, dtype_loc):
            if dtype in 'ifb':
                v[:, i] = self._data[dtype][:, loc]
                i += 1
        return v

    def _values_raw(self, kinds: Set[str]) -> ndarray:
        """
        Retrieve all the DataFrame values into an array. Booleans will be coerced to ints
        """
        if 'O' in self._data and 'O' in kinds:
            arr_dtype: str = 'O'
        elif 'f' in self._data and 'f' in kinds:
            arr_dtype = 'float64'
        else:
            arr_dtype = 'int64'

        col_num: int = 0
        for kind, arr in self._data.items():
            if kind in kinds:
                col_num += arr.shape[1]
        shape = (len(self), col_num)

        v: ndarray = np.empty(shape, dtype=arr_dtype, order='F')
        i = 0
        for col in self._columns:
            dtype, loc, order = self._column_info[col].values  # type: str, int, int
            if dtype in kinds:
                v[:, i] = self._data[dtype][:, loc]
                i += 1
        return v

    def _get_column_values(self, col: str) -> ndarray:
        """
        Retrieve a 1d array of a single column
        """
        dtype, loc, _ = self._column_info[col].values
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
                else:
                    data = [column] + vals.tolist()
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

            if self._column_info[column].dtype == 'O':
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

                if isinstance(d, (float, np.floating)) and np.isnan(d):
                    d = 'NaN'
                if isinstance(d, bool):
                    d = str(d)
                if d is None:
                    d = 'None'

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
            for j, (d, fl, dl) in enumerate(zip(data_list, long_len,
                                                decimal_len)):
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
        return len(self), len(self._columns)

    @property
    def size(self) -> int:
        return len(self) * len(self._columns)

    def _find_col_location(self, col: str) -> int:
        try:
            return self._column_info[col].order
        except KeyError:
            raise KeyError(f'{col} is not in the columns')

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
                                'You must only use booleans by themselves.')

        if bool_count > 0:
            if bool_count != self.shape[1]:
                raise ValueError('The length of the boolean list must match the number of '
                                 f'columns {i} != {self.shape[1]}')

        utils.check_duplicate_list(new_cols)
        return new_cols

    def _convert_col_selection(self, cs: ColSelection) -> List[str]:
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

    def _convert_row_selection(self, rs: RowSelection) -> Union[List[int], ndarray]:
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
            row_array = rs.values.squeeze()
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
        elif not isinstance(rs, int):
            raise TypeError('Selection must either be one of '
                            'int, list, array, slice, or DataFrame')
        return rs

    def _getitem_scalar(self, rs: int, cs: Union[int, str]) -> Scalar:
        # most common case, string column, integer row
        try:
            dtype, loc, _ = self._column_info[cs].values  # type: ignore
            return self._data[dtype][rs, loc]
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
                dtype, loc, _ = self._column_info[cs].values  # type: ignore
                return self._data[dtype][rs, loc]

    def _select_entire_single_column(self, col_selection):
        self._validate_column_name(col_selection)
        dtype, loc, _ = self._column_info[col_selection].values
        new_data = {dtype: self._data[dtype][:, [loc]]}
        new_columns = [col_selection]
        new_column_info = {col_selection: utils.Column(dtype, 0, 0)}
        return self._construct_from_new(new_data, new_column_info, new_columns)

    def _construct_df_from_selection(self, rs: Union[List[int], ndarray],
                                     cs: List[str]) -> 'DataFrame':
        new_data: Dict[str, ndarray] = {}
        dt_positions: Dict[str, List[int]] = defaultdict(list)
        new_column_info: ColInfoT = {}
        new_columns: ColumnT = cs

        col: str
        for i, col in enumerate(cs):
            dtype, loc, _ = self._column_info[col].values
            cur_loc: int = len(dt_positions[dtype])
            new_column_info[col] = utils.Column(dtype, cur_loc, i)
            dt_positions[dtype].append(loc)

        for dtype, pos in dt_positions.items():

            if isinstance(rs, (list, ndarray)):
                ix = np.ix_(rs, pos)
                arr = np.atleast_2d(self._data[dtype][ix])
            else:
                arr = np.atleast_2d(self._data[dtype][rs, pos])

            new_data[dtype] = np.asfortranarray(arr)

        return self._construct_from_new(new_data, new_column_info, new_columns)

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

    def __getitem__(self, value: Tuple[RowSelection,
                                       ColSelection]) -> Union[Scalar, 'DataFrame']:
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
        row_selection, col_selection = value  # type: RowSelection, ColSelection
        if isinstance(row_selection, int) and isinstance(col_selection, (int, str)):
            return self._getitem_scalar(row_selection, col_selection)

        if utils.is_entire_column_selection(row_selection, col_selection):
            return self._select_entire_single_column(col_selection)

        cs_final: List[str] = self._convert_col_selection(col_selection)
        rs_final: Union[List[int], ndarray] = self._convert_row_selection(row_selection)

        return self._construct_df_from_selection(rs_final, cs_final)

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
        col_obj: utils.Column
        for col in self._columns:
            dtype, loc, order = self._column_info[col].values  # type: str, int, int
            arr = self._data[dtype][:, loc]
            if orient == 'array':
                data[col] = arr.copy()
            else:
                data[col] = arr.tolist()
        return data

    def _is_numeric_or_bool(self) -> bool:
        return set(self._data.keys()) <= set('bif')

    def _is_numeric_strict(self) -> bool:
        return set(self._data.keys()) <= {'i', 'f'}

    def _is_string(self) -> bool:
        return set(self._data.keys()) == {'O'}

    def _is_date(self) -> bool:
        return set(self._data.keys()) <= {'m', 'M'}

    def _is_only_numeric_or_string(self) -> bool:
        dtypes: Set[str] = set(self._data.keys())
        return dtypes <= {'i', 'f'} or dtypes == {'O'}

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
        dtypes: Set[str] = set(self._data.keys())
        return 'i' in dtypes or 'f' in dtypes

    def _has_string(self) -> bool:
        return 'O' in self._data

    def copy(self) -> 'DataFrame':
        """
        Returns an exact replica of the DataFrame as a copy

        Returns
        -------
        A copy of the DataFrame

        """
        new_data: Dict[str, ndarray] = {dt: arr.copy() for dt, arr in self._data.items()}
        new_columns: ColumnT = self._columns.copy()
        new_column_info: ColInfoT = {col: utils.Column(*col_obj.values)
                                     for col, col_obj in self._column_info.items()}
        return self._construct_from_new(new_data, new_column_info, new_columns)

    def select_dtypes(self, include: Optional[Union[str, List[str]]] = None,
                      exclude: Optional[Union[str, List[str]]] = None) -> 'DataFrame':
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

        clude: Union[str, List[str]]
        if include is None:
            clude = exclude
            arg_name: str = 'exclude'
        if exclude is None:
            clude = include
            arg_name = 'include'

        clude_final: Union[str, List[str]] = utils.convert_clude(clude, arg_name)

        include_final: List[str]
        current_dtypes = set(self._data.keys())
        if arg_name == 'include':
            include_final = [dt for dt in current_dtypes if dt in clude_final]
        else:
            include_final = [dt for dt in current_dtypes if dt not in clude_final]

        new_data: Dict[str, ndarray] = {dtype: arr.copy('F')
                                        for dtype, arr in self._data.items()
                                        if dtype in include_final}

        new_column_info: ColInfoT = {}
        new_columns: ColumnT = []
        order: int = 0
        for col in self._columns:
            dtype, loc, _ = self._column_info[col].values  # type: str, int, int
            if dtype in include_final:
                new_column_info[col] = utils.Column(dtype, loc, order)
                new_columns.append(col)
                order += 1
        return self._construct_from_new(new_data, new_column_info, new_columns)

    @classmethod
    def _construct_from_new(cls: Type[object], data: Dict[str, ndarray],
                            column_info: ColInfoT, columns: ColumnT) -> 'DataFrame':
        df_new: 'DataFrame' = super().__new__(cls)
        df_new._column_info = column_info
        df_new._data = data
        df_new._columns = np.asarray(columns, dtype='O')
        df_new._hasnans = {}
        df_new._add_accessors()
        return df_new

    def _do_eval(self, op_string: str, other: Any) -> Tuple[Dict[str, List[ndarray]], ColInfoT]:
        data_dict: Dict[str, List[ndarray]] = defaultdict(list)
        new_column_info: ColInfoT = {}
        kind_shape: Dict[str, Tuple[str, int]] = {}
        arr_res: ndarray
        new_kind: str
        new_loc: int
        cur_len: int
        old_kind: str
        old_loc: int
        order: int

        for old_kind, arr in self._data.items():
            with np.errstate(invalid='ignore', divide='ignore'):
                # TODO: will have to do custom cython function here for add,
                # radd, gt, ge, lt, le for object array
                if old_kind == 'O' and op_string in stat.funcs_str:
                    func = stat.funcs_str[op_string]
                    arr_res = func(arr, other)
                else:
                    arr_res = eval(f"{'arr'} .{op_string}({'other'})")
            new_kind = arr_res.dtype.kind

            cur_len = utils.get_num_cols(data_dict.get(new_kind, []))
            kind_shape[old_kind] = (new_kind, cur_len)
            data_dict[new_kind].append(arr_res)

        for col, col_obj in self._column_info.items():
            old_kind, old_loc, order = col_obj.values
            new_kind, new_loc = kind_shape[old_kind]
            new_column_info[col] = utils.Column(new_kind, new_loc + old_loc, order)
        return data_dict, new_column_info

    def _get_both_column_info(self, other: 'DataFrame') -> Tuple:
        kinds1: List[str] = []
        kinds2: List[str] = []
        locs1: Dict[str, List[int]] = defaultdict(list)
        locs2: Dict[str, List[int]] = defaultdict(list)
        ords1: Dict[str, List[int]] = defaultdict(list)
        ords2: Dict[str, List[int]] = defaultdict(list)
        cols1: Dict[str, List[str]] = defaultdict(list)
        cols2: Dict[str, List[str]] = defaultdict(list)

        columns1, columns2 = self._columns, other._columns  # type: ndarray, ndarray
        if len(columns1) > len(columns2):
            columns2 = columns2.repeat(len(columns1))
        elif len(columns1) < len(columns2):
            columns1 = columns1.repeat(len(columns2))

        for col1, col2 in zip(columns1, columns2):  # type: str, str
            dtype1, loc1, order1 = self._column_info[col1].values
            dtype2, loc2, order2 = other._column_info[col2].values
            kinds1.append(dtype1)
            kinds2.append(dtype2)
            locs1[dtype1].append(loc1)
            locs2[dtype2].append(loc2)
            ords1[dtype1].append(order1)
            ords2[dtype2].append(order2)
            cols1[dtype1].append(col1)
            cols2[dtype2].append(col2)
        return kinds1, kinds2, locs1, locs2, ords1, ords2, cols1, cols2

    def _get_single_column_values(self, iloc: int) -> ndarray:
        col = self._columns[iloc]
        dtype, loc, order = self._column_info[col].values  # type: str, int, int
        return self._data[dtype][:, loc]

    def _op(self, other: Any, op_string: str) -> 'DataFrame':
        if isinstance(other, (int, float, bool)):
            if self._is_numeric_or_bool() or op_string in ['__mul__', '__rmul__']:
                dd, ncd = self._do_eval(op_string, other)
            else:
                raise TypeError('This operation only works when all the columns are numeric.')
        elif isinstance(other, str):
            if self._is_string():
                if op_string in ['__add__', '__radd__', '__gt__', '__ge__',
                                 '__lt__', '__le__', '__ne__', '__eq__']:
                    dd, ncd = self._do_eval(op_string, other)
                else:
                    special_method_name = utils.convert_special_method(op_string)
                    raise TypeError(f'Cannot do {special_method_name} with string types')
            else:
                raise TypeError('All columns in DataFrame must be string if '
                                'operating with a string')
        elif isinstance(other, np.timedelta64):
            dtypes = set(self._data.keys())
            if dtypes == {'M', 'm'} or dtypes == {'M'}:
                if op_string in ['__add__', '__radd__', '__sub__']:
                    dd, ncd = self._do_eval(op_string, other)
                else:
                    special_method_name = utils.convert_special_method(op_string)
                    raise TypeError(f'Cannot do {special_method_name} with datetime/timedelta'
                                    f' types')
            elif dtypes == {'m'}:
                if op_string in ['__add__', '__radd__', '__gt__', '__ge__',
                                 '__lt__', '__le__', '__ne__', '__eq__']:
                    dd, ncd = self._do_eval(op_string, other)
                else:
                    special_method_name = utils.convert_special_method(op_string)
                    raise TypeError(f'Cannot do {special_method_name} with timedelta types')
            else:
                raise TypeError('When operating with a timedelta, all columns in the DataFrame '
                                'must be either datetime64 or timedelta64 ')
        elif isinstance(other, np.datetime64):
            dtypes = set(self._data.keys())
            if dtypes == {'M'}:
                if op_string in ['__sub__', '__gt__', '__ge__', '__lt__',
                                 '__le__', '__ne__', '__eq__']:
                    dd, ncd = self._do_eval(op_string, other)
                else:
                    special_method_name = utils.convert_special_method(op_string)
                    raise TypeError(f'Cannot do {special_method_name} with datetime/timedelta'
                                    f' types')
            elif dtypes == {'m'}:
                if op_string in ['__add__', '__radd__', '__sub__', '__rsub__']:
                    dd, ncd = self._do_eval(op_string, other)
                else:
                    special_method_name = utils.convert_special_method(op_string)
                    raise TypeError(f'Cannot do {special_method_name} with string types')
            elif dtypes == {'M', 'm'}:
                if op_string in ['__sub__', '__rsub__']:
                    dd, ncd = self._do_eval(op_string, other)
                else:
                    special_method_name = utils.convert_special_method(op_string)
                    raise TypeError(f'Cannot do {special_method_name} with string types')
            else:
                raise TypeError('When operating with a timedelta, all columns in the DataFrame '
                                'must be either datetime64 or timedelta64 ')
        elif isinstance(other, DataFrame):
            def get_cur_arr(self, other: 'DataFrame', dtype1: str, dtype2: str,
                            locs1: Dict[str, List[int]],
                            locs2: Dict[str, List[int]]):
                data1: ndarray = self._data[dtype1]
                data2: ndarray = other._data[dtype2]
                cur_locs1, cur_locs2 = locs1[dtype1], locs2[dtype2]  # type: List[int], List[int]
                if cur_locs1 != cur_locs2:
                    data1 = data1[:, cur_locs1]
                    data2 = data2[:, cur_locs2]
                    cur_locs1 = list(range(len(cur_locs1)))
                # TODO: multiply string by number dataframe. very rare occurrence
                if dtype1 == 'O':
                    if self.shape == other.shape and op_string in stat.funcs_str2:
                        func: Callable = stat.funcs_str2[op_string]
                        return func(data1, data2), cur_locs1
                    elif (self.shape[0] == 1 or other.shape[
                        0] == 1) and op_string in stat.funcs_str2_bc:

                        func = stat.funcs_str2_bc[op_string]
                        return func(data1, data2), cur_locs1
                return eval(f"{'data1'} .{op_string}({'data2'})"), cur_locs1

            if self.shape == other.shape or (self.shape[1] == other.shape[1] and (
                    self.shape[0] == 1 or other.shape[0] == 1)):
                kinds1: List[str]
                kinds2: List[str]
                locs1: Dict[str, List[int]]
                locs2: Dict[str, List[int]]
                ords1: Dict[str, List[int]]
                ords2: Dict[str, List[int]]
                cols1: Dict[str, List[str]]
                cols2: Dict[str, List[str]]

                kinds1, kinds2, locs1, locs2, ords1, ords2, cols1, cols2 = \
                    self._get_both_column_info(other)
                data_dict: Dict[str, List[ndarray]] = defaultdict(list)
                new_column_info: ColInfoT = {}
                new_columns: ndarray = self._columns.copy()
                if kinds1 == kinds2:
                    # fast path for similar data frames
                    for dtype1 in self._data:
                        arr_new, cur_locs1 = get_cur_arr(self, other, dtype1, dtype1, locs1,
                                                         locs2)  # type: ndarray, List[int]
                        new_dtype: str = arr_new.dtype.kind
                        cur_len: int = utils.get_num_cols(data_dict.get(new_dtype, []))
                        data_dict[new_dtype].append(arr_new)
                        old_info: Iterable = zip(cols1[dtype1], cur_locs1, ords1[dtype1])
                        for col, loc, order in old_info:  # type: str, int, int
                            new_column_info[col] = utils.Column(new_dtype, loc + cur_len, order)

                    new_data: Dict[str, ndarray] = utils.concat_stat_arrays(data_dict)
                    return self._construct_from_new(new_data, new_column_info, new_columns)
                elif utils.check_compatible_kinds(kinds1, kinds2, [False] * len(kinds1)):
                    # fast path for single dtype frames
                    if len(set(kinds1)) == 1 and len(set(kinds2)) == 1:
                        dtype1, dtype2 = kinds1[0], kinds2[0]
                        arr_new, cur_locs1 = get_cur_arr(self, other, dtype1, dtype2, locs1, locs2)
                        new_dtype = arr_new.dtype.kind
                        new_data = {new_dtype: arr_new}
                        old_info = zip(cols1[dtype1], cur_locs1, ords1[dtype1])
                        for col, loc, order in old_info:
                            new_column_info[col] = utils.Column(new_dtype, loc, order)
                        return self._construct_from_new(new_data, new_column_info, new_columns)
                    else:
                        # different kinds but compatible and more than one dtype
                        new_data = {}
                        if 'O' in self._data:
                            arr_new, cur_locs1 = get_cur_arr(self, other, 'O', 'O', locs1, locs2)
                            new_dtype = arr_new.dtype.kind
                            new_data[new_dtype] = arr_new

                            for col, loc, order in zip(cols1['O'], cur_locs1, ords1['O']):
                                new_column_info[col] = utils.Column('O', loc, order)

                        col_arrs1: Dict[str, List[ndarray]] = defaultdict(list)
                        col_arrs2: Dict[str, List[ndarray]] = defaultdict(list)

                        for i, (kind1, kind2) in enumerate(zip(kinds1, kinds2)):
                            if kind1 == 'O':
                                continue
                            col = self._columns[i]
                            col_arr1 = self._get_single_column_values(i)
                            col_arr2 = other._get_single_column_values(i)
                            if (kind1 == 'f' and kind2 in 'fib') or (
                                    kind2 == 'f' and kind1 in 'fib'):
                                new_kind = 'f'
                            elif (kind1 == 'i' and kind2 in 'ib') or (
                                    kind2 == 'i' and kind1 in 'ib'):
                                new_kind = 'i'
                            else:
                                new_kind = 'b'

                            cur_loc = len(col_arrs1[new_kind])
                            new_column_info[col] = utils.Column(new_kind, cur_loc, i)
                            col_arrs1[new_kind].append(col_arr1)
                            col_arrs2[new_kind].append(col_arr2)

                        for dtype, arrs1 in col_arrs1.items():
                            arrs2 = col_arrs2[dtype]
                            if len(arrs1) == 1:
                                data1, data2 = arrs1[0][:, np.newaxis], arrs2[0][:, np.newaxis]
                            else:
                                data1, data2 = np.column_stack(arrs1), np.column_stack(arrs2)
                            arr_new = eval(f"{'data1'} .{op_string}({'data2'})")
                            new_data[dtype] = arr_new
                        return self._construct_from_new(new_data, new_column_info, new_columns)
                else:
                    for i, (kind1, kind2) in enumerate(zip(kinds1, kinds2)):
                        if kind1 == 'O' and kind1 != 'O':
                            break
                        if kind1 in 'ifb' and kind2 not in 'ifb':
                            break
                    raise ValueError(f'Column {self._columns[i]} has an incompatible type '
                                     f'with column {other._columns[i]}')
            elif self.shape[0] == other.shape[0]:
                ncol_self, ncol_other = self.shape[1], other.shape[1]  # type: int, int
                if ncol_self == 1 or ncol_other == 1:
                    kinds1, kinds2, locs1, locs2, ords1, ords2, cols1, cols2 = \
                        self._get_both_column_info(other)

                    larger_df: 'DataFrame'
                    smaller_df: 'DataFrame'
                    cols_final: Dict[str, List[str]]
                    locs_final: Dict[str, List[int]]
                    ords_final: Dict[str, List[int]]
                    ncol_larger: int

                    if ncol_self > ncol_other:
                        larger_df, cols_final, locs_final, ords_final = self, cols1, locs1, ords1
                        smaller_df = other
                        ncol_larger = ncol_self
                    else:
                        larger_df, cols_final, locs_final, ords_final = other, cols2, locs2, ords2
                        ncol_larger = ncol_other
                        smaller_df = self

                    utils.check_compatible_kinds(kinds1 * ncol_other, kinds2 * ncol_self,
                                                 [False] * ncol_larger)
                    data_dict = defaultdict(list)
                    new_column_info = {}
                    for dtype, data1 in larger_df._data.items():
                        data2 = list(smaller_df._data.values())[0]
                        if dtype == 'O' and op_string in stat.funcs_str2_bc:
                            func = stat.funcs_str2_bc[op_string]
                            arr_new = func(data1, data2)
                        else:
                            arr_new = eval(f"{'data1'} .{op_string}({'data2'})")
                        new_dtype = arr_new.dtype.kind
                        cur_loc = utils.get_num_cols(data_dict[new_dtype])
                        data_dict[new_dtype].append(arr_new)
                        old_info = zip(cols_final[dtype], locs_final[dtype], ords_final[dtype])
                        for col, loc, order in old_info:
                            new_column_info[col] = utils.Column(new_dtype, cur_loc + loc, order)

                    new_data = utils.concat_stat_arrays(data_dict)
                    new_columns = larger_df._columns.copy()
                    return self._construct_from_new(new_data, new_column_info, new_columns)
                else:
                    raise ValueError('Both DataFrames have the same number of rows but a '
                                     'different number of columns. To make this operation work, '
                                     'both DataFrames need the same shape or have one axis the '
                                     'same and the other length of 1.')
            elif self.shape[1] == other.shape[1]:
                if self.shape[0] == 1 or other.shape[0] == 1:
                    pass
                else:
                    raise ValueError('Both DataFrames have the same number of columns but a '
                                     'different number of rows. To make this operation work, '
                                     'both DataFrames need the same shape or have one axis the '
                                     'same and the other length of 1.')
            else:
                raise ValueError('Both DataFrames have a different number of rows and columns. '
                                 'To make this operation work, '
                                 'both DataFrames need the same shape or have one axis the '
                                 'same and the other length of 1.')
        elif isinstance(other, ndarray):
            if other.ndim == 1:
                other = other[:, np.newaxis]
            elif other.ndim != 2:
                raise ValueError('array must have one or two dimensions')

            new_data = {}
            dtype = other.dtype.kind
            if dtype == 'U':
                dtype = 'O'
                new_data['O'] = other
            elif dtype in 'Oifb':
                new_data[dtype] = other
            else:
                raise ValueError('Unknown array data type')

            if dtype == 'O':
                other = DataFrame(other)
            else:
                new_columns = ['a' + str(i) for i in range(other.shape[1])]
                new_column_info = {col: utils.Column(dtype, i, i) for i, col in
                                   enumerate(new_columns)}
                other = self._construct_from_new(new_data, new_column_info, new_columns)
            return eval(f"{'self'} .{op_string}({'other'})")
        else:
            raise TypeError('other must be int, float, str, bool, timedelta, '
                            'datetime, array or DataFrame')

        new_data = {}
        for dt, arrs in dd.items():
            if len(arrs) == 1:
                new_data[dt] = arrs[0]
            else:
                new_data[dt] = np.concatenate(arrs, axis=1)

        return self._construct_from_new(new_data, ncd, self._columns.copy())

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

    def __eq__(self, other: Any) -> 'DataFrame':  # type: ignore
        return self._op(other, '__eq__')

    def __ne__(self, other: Any) -> 'DataFrame':  # type: ignore
        return self._op(other, '__ne__')

    def __neg__(self) -> 'DataFrame':
        if self._is_numeric_or_bool():
            new_data: Dict[str, ndarray] = {}
            for dt, arr in self._data.items():
                new_data[dt] = -arr
        else:
            raise TypeError('Only works for all numeric columns')
        new_column_info: ColInfoT = {col: utils.Column(*col_obj.values)
                                     for col, col_obj in self._column_info.items()}
        return self._construct_from_new(new_data, new_column_info, self._columns.copy())

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
        cn: ndarray = self._columns.astype('O')
        data: ndarray = np.column_stack((cn, arr))
        new_data: Dict[str, ndarray] = {'O': data}
        new_column_info: ColInfoT = {'Column Name': utils.Column('O', 0, 0),
                                     'Data Type': utils.Column('O', 1, 1)}
        return self._construct_from_new(new_data, new_column_info, columns)

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
        is_other_df = isinstance(other, DataFrame)
        new_data: Dict[str, ndarray] = {}

        arr_left: ndarray = self.values
        arr_right: ndarray

        if is_other_df:
            arr_right = other.values
        else:
            arr_right = other

        arr_res: ndarray = getattr(arr_left, op_logical)(arr_right)
        new_data[arr_res.dtype.kind] = arr_res
        new_column_info: ColInfoT = {col: utils.Column('b', i, i)
                                     for i, (col, _) in
                                     enumerate(self._column_info.items())}

        return self._construct_from_new(new_data, new_column_info, self._columns.copy())

    def __invert__(self) -> 'DataFrame':
        if set(self._data) == {'b'}:
            new_data: Dict[str, ndarray] = {dt: ~arr for dt, arr in self._data.items()}
            new_column_info: ColInfoT = self._copy_column_info()
            return self._construct_from_new(new_data, new_column_info, self._columns.copy())
        else:
            raise TypeError('Invert operator only works on DataFrames with all boolean columns')

    def _copy_column_info(self) -> ColInfoT:
        return {col: utils.Column(*col_obj.values)
                for col, col_obj in self._column_info.items()}

    def _astype_internal(self, column: str, numpy_dtype: str) -> None:
        """
        Changes one column dtype in-place
        """
        new_kind: str = utils.convert_numpy_to_kind(numpy_dtype)
        dtype, loc, order = self._column_info[column].values  # type: str, int, int

        if dtype == new_kind:
            return None
        col_data: ndarray = self._data[dtype][:, loc]
        if numpy_dtype == 'O':
            if dtype in 'mM':
                nulls: ndarray = np.isnat(col_data)
            else:
                nulls = np.isnan(col_data)
            col_data = col_data.astype('U').astype('O')
            col_data[nulls] = None
        else:
            col_data = col_data.astype(numpy_dtype)

        if col_data.dtype.kind == 'M':
            col_data = col_data.astype('datetime64[ns]')
        elif col_data.dtype.kind == 'm':
            col_data = col_data.astype('timedelta64[ns]')

        self._remove_column(column)
        self._write_new_column_data(column, new_kind, col_data, order)

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

    def _write_new_column_data(self, column: str, new_kind: str, data: ndarray, order: int) -> None:
        """
        Adds data to _data, the data type info but does not
        append name to columns
        """
        if new_kind not in self._data:
            loc = 0
        else:
            loc = self._data[new_kind].shape[1]
        self._column_info[column] = utils.Column(new_kind, loc, order)
        if new_kind in self._data:
            self._data[new_kind] = np.asfortranarray(np.column_stack((self._data[new_kind], data)))
        else:
            if data.ndim == 1:
                data = data[:, np.newaxis]

            self._data[new_kind] = np.asfortranarray(data)

    def _add_new_column(self, column: str, kind: str, data: ndarray) -> None:
        order: int = len(self._columns)
        self._write_new_column_data(column, kind, data, order)
        self._columns = np.append(self._columns, column)

    def _full_columm_add(self, column: str, kind: str, data: ndarray) -> None:
        """
        Either adds a brand new column or
        overwrites an old column
        """
        if column not in self._column_info:
            self._add_new_column(column, kind, data)
        # column is in df
        else:
            # data type has changed
            dtype, loc, order = self._column_info[column].values  # type: str, int, int
            if dtype != kind:
                self._remove_column(column)
                self._write_new_column_data(column, kind, data, order)
            # data type same as original
            else:
                self._data[kind][:, loc] = data

    def __setitem__(self, key: Any, value: Any) -> None:
        utils.validate_selection_size(key)

        # row selection and column selection
        rs: RowSelection
        cs: ColSelection
        rs, cs = key
        if isinstance(rs, int) and isinstance(cs, (int, str)):
            return self._setitem_scalar(rs, cs, value)

        # select an entire column or new column
        if utils.is_entire_column_selection(rs, cs):
            # ignore type check here. Function ensures cs is a string
            return self._setitem_entire_column(cs, value)  # type: ignore

        col_list: List[str] = self._convert_col_selection(cs)
        row_list: Union[List[int], ndarray] = self._convert_row_selection(rs)

        self._setitem_all_other(row_list, col_list, value)

    def _setitem_scalar(self, rs: RowSelection, cs: Union[int, str], value: Scalar) -> None:
        """
        Assigns a scalar to exactly a single cell
        """
        if isinstance(cs, str):
            self._validate_column_name(cs)
            col_name: str = cs
        else:
            col_name = self._get_col_name_from_int(cs)
        dtype, loc, order = self._column_info[col_name].values  # type: str, int, int

        if isinstance(value, bool):
            utils.check_set_value_type(dtype, 'b', 'bool')
        elif isinstance(value, np.datetime64):
            utils.check_set_value_type(dtype, 'M', 'datetime64')
        elif isinstance(value, np.timedelta64):
            utils.check_set_value_type(dtype, 'm', 'timedelta64')
        elif isinstance(value, (int, np.integer)):
            utils.check_set_value_type(dtype, 'if', 'int')
        elif isinstance(value, (float, np.floating)):
            if np.isnan(value):
                if dtype in 'ib':
                    self._astype_internal(col_name, 'float64')
                    dtype = 'f'
                    loc = -1
                elif dtype == 'O':
                    raise ValueError("Can't set nan to a str column. Use `None` instead.")
            else:
                utils.check_set_value_type(dtype, 'if', 'float')
                if dtype == 'i':
                    self._astype_internal(col_name, 'float64')
                    dtype = 'f'
                    loc = -1
        elif isinstance(value, str):
            utils.check_set_value_type(dtype, 'O', 'str')
        elif value is None:
            utils.check_set_value_type(dtype, 'O', 'None')
        else:
            raise TypeError(f'Type {type(value).__name__} not able to be assigned')
        self._data[dtype][rs, loc] = value

    def _setitem_entire_column(self, cs: str, value: Union[Scalar, ndarray, 'DataFrame']) -> None:
        """
        Called when setting an entire column (old or new)
        df[:, 'col'] = value
        """
        if utils.is_scalar(value):
            data: ndarray = np.repeat(value, len(self))
            data = utils.convert_bytes_or_unicode(data)
            kind: str = data.dtype.kind
            self._full_columm_add(cs, kind, data)
        elif isinstance(value, (ndarray, list)):
            if isinstance(value, list):
                value = utils.convert_list_to_single_arr(value)
                value = utils.maybe_convert_1d_array(value)
            value = utils.try_to_squeeze_array(value)
            utils.validate_array_size(value, len(self))
            if value.dtype.kind == 'U':
                value = value.astype('O')
            if value.dtype.kind == 'O':
                _va.validate_strings_in_object_array(value)
            self._full_columm_add(cs, value.dtype.kind, value)
        elif isinstance(value, DataFrame):
            if value.shape[0] != self.shape[0]:
                raise ValueError(f'The DataFrame on the left has {self.shape[0]} rows. '
                                 f'The DataFrame on the right has {self.shape[0]} rows. '
                                 'They must be equal')
            if value.shape[1] != 1:
                raise ValueError('You are setting exactly one column. The DataFrame you are '
                                 f'trying to set this with has {value.shape[1]} columns. '
                                 'They must be equal')
            data = value.values.squeeze()
            kind = data.dtype.kind
            self._full_columm_add(cs, kind, data)
        else:
            raise TypeError('Must use a scalar, a list, an array, or a '
                            'DataFrame when setting new values')

    def _setitem_all_other(self, rs: Union[List[int], ndarray], cs: List[str], value: Any) -> None:
        """
        Sets new data when not assigning a scalar
        and not assigning a single column
        """
        value_kind: Optional[str] = utils.get_kind_from_scalar(value)
        if value_kind:
            self._validate_setitem_col_types(cs, value_kind)
            for col in cs:
                dtype, loc, order = self._column_info[col].values  # type: str, int, int
                if dtype == 'i' and value_kind == 'f':
                    self._astype_internal(col, 'float64')
                    dtype, loc, order = self._column_info[col].values
                self._data[dtype][rs, loc] = value
        # not scalar
        else:
            nrows_to_set, ncols_to_set = self._get_nrows_cols_to_set(rs, cs)  # type: int, int
            single_row: bool = utils.is_one_row(nrows_to_set, ncols_to_set)

            arrs: List[ndarray]
            if isinstance(value, list):
                arrs = utils.convert_list_to_arrays(value, single_row)
            elif isinstance(value, ndarray):
                arrs = utils.convert_array_to_arrays(value)
            elif isinstance(value, DataFrame):
                arrs = []
                for col, col_obj in value._column_info.items():
                    dtype, loc, order = col_obj.values
                    arrs.append(value._data[dtype][:, loc])
            else:
                raise TypeError('Must use a scalar, a list, an array, or a '
                                'DataFrame when setting new values')
            self._validate_set_array_shape(nrows_to_set, ncols_to_set, arrs)
            # need to check for object nan compatibility
            kinds, other_kinds = self._get_kinds(cs, arrs)  # type: List[str], List[str]
            all_nans: List[bool] = utils.check_all_nans(arrs)
            # recently included compatibility of booleans with ints and floats
            utils.check_compatible_kinds(kinds, other_kinds, all_nans)

            # Must use scalar value when setting object arrays
            if single_row:
                arrs = [arr[0] for arr in arrs]
            self._setitem_other_cols(rs, cs, arrs, kinds, other_kinds)

    def _get_nrows_cols_to_set(self, rs: Union[List[int], ndarray],
                               cs: List[str]) -> Tuple[int, int]:
        if isinstance(rs, int):
            rows: int = 1
        else:
            rows = len(np.arange(len(self))[rs])
        cols: int = len(cs)
        return rows, cols

    def _validate_setitem_col_types(self, columns: List[str], kind: str) -> None:
        """
        Used to verify column dtypes when setting a scalar
        to many columns
        """
        for col in columns:
            cur_kind: str = self._column_info[col].dtype
            if cur_kind == kind or (cur_kind in 'if' and kind in 'if'):
                continue
            else:
                dt: str = utils.convert_kind_to_dtype(kind)
                ct: str = utils.convert_kind_to_dtype(cur_kind)
                raise TypeError(f'Trying to set a {dt} on column {col} which has type {ct}')

    def _validate_set_array_shape(self, nrows_to_set: int, ncols_to_set: int,
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

    def _get_kinds(self, cols: List[str],
                   other: Union[List[ndarray], 'DataFrame']) -> Tuple[List[str], List[str]]:
        kinds: List[str] = [self._column_info[col].dtype for col in cols]
        if isinstance(other, list):
            other_kinds: List[str] = [arr.dtype.kind for arr in other]
        else:
            other_kinds = [other._column_info[col].dtype
                           for col in other._columns]
        return kinds, other_kinds

    def _setitem_other_cols(self, rows: Union[List[int], ndarray], cols: List[str],
                            arrs: List[ndarray], kinds1: List[str], kinds2: List[str]) -> None:
        for col, arr, k1, k2 in zip(cols, arrs, kinds1, kinds2):  # type: str, ndarray, str, str
            if k1 == 'i' and k2 == 'f':
                dtype_internal: str = utils.convert_kind_to_numpy(k2)
                self._astype_internal(col, dtype_internal)
            dtype, loc, order = self._column_info[col].values  # type: str, int, int
            self._data[dtype][rows, loc] = arr

    def _validate_column_name(self, column: str) -> None:
        if column not in self._column_info:
            raise KeyError(f'Column {column} does not exist')

    def _validate_column_name_list(self, columns: list) -> None:
        col_set: Set[str] = set()
        for col in columns:
            self._validate_column_name(col)
            if col in col_set:
                raise ValueError(f'"{col}" has already been selected as a column')
            col_set.add(col)

    def _new_cd_from_kind_shape(self, kind_shape: Dict[str, int], new_kind: str) -> ColInfoT:
        new_column_info: ColInfoT = {}
        for col, col_obj in self._column_info.items():
            dtype, loc, order = col_obj.values  # type: str, int, int
            add_loc: int = kind_shape[dtype]
            new_column_info[col] = utils.Column(new_kind, loc + add_loc,
                                                order)
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
            data_dict: Dict[str, List[ndarray]] = defaultdict(list)
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
                    elif old_kind == 'O':
                        nanable = True
                        na_arr = _math.isna_str(arr, np.zeros(len(arr), dtype='bool'))
                    elif old_kind in 'mM':
                        nanable = True
                        na_arr = np.isnat(arr)

                    if new_dtype == 'O':
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

            new_column_info: ColInfoT = self._new_cd_from_kind_shape(kind_shape, new_kind)
            new_data: Dict[str, ndarray] = utils.concat_stat_arrays(data_dict)
            return self._construct_from_new(new_data, new_column_info, self._columns.copy())

        elif isinstance(dtype, dict):
            df_new: 'DataFrame' = self.copy()

            column: str
            new_dtype2: str
            for column, new_dtype2 in dtype.items():
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
            kind: str
            arr: ndarray
            for kind, arr in self._data.items():
                if kind in 'f':
                    self._hasnans['f'] = np.isnan(arr).any(0)
                elif kind == 'O':
                    self._hasnans['O'] = _va.isnan_object(arr)
                elif kind in 'mM':
                    self._hasnans[kind] = np.isnat(arr).any(0)
                else:
                    self._hasnans[kind] = np.zeros(arr.shape[1], dtype='bool')

        bool_array: ndarray = np.empty(len(self), dtype='bool')

        col: str
        col_obj: utils.Column
        for col, col_obj in self._column_info.items():
            kind2, loc, order = col_obj.values  # type: str, int, int
            bool_array[order] = self._hasnans[kind2][loc]

        columns: ndarray = np.array(['Column Name', 'Has NaN'])
        new_data: Dict[str, ndarray] = {'O': self._columns[:, np.newaxis],
                                        'b': bool_array[:, np.newaxis]}

        new_column_info: ColInfoT = {'Column Name': utils.Column('O', 0, 0),
                                     'Has NaN': utils.Column('b', 0, 1)}
        return self._construct_from_new(new_data, new_column_info, columns)

    def _get_specific_stat_dtypes(self, name: str, axis: int) -> Set[str]:
        if self._is_string():
            if name in ['std', 'var', 'mean', 'median', 'quantile']:
                raise TypeError('Your DataFrame consists entirely of strings. '
                                f'The `{name}` method only works with numeric columns.')
            return {'O'}
        if axis == 0:
            if name in ['std', 'var', 'mean', 'median', 'quantile', 'prod', 'cumprod']:
                return {'i', 'f', 'b'}
        elif axis == 1:
            if name in ['std', 'var', 'mean', 'median', 'quantile', 'sum', 'max', 'min', 'mode',
                        'cumsum', 'cummax', 'cummin', 'argmin', 'argmax', 'prod', 'cumprod']:
                return {'i', 'f', 'b'}

        if name in ['max', 'min', 'any', 'all', 'argmin', 'argmax', 'count',
                    'cummax', 'cummin', 'nunique']:
            return {'i', 'f', 'b', 'O', 'm', 'M'}
        else:
            return {'i', 'f', 'b', 'O'}

    def _get_stat_func_result(self, func: Callable, arr: ndarray,
                              axis: int, kwargs: Dict) -> ndarray:
        result: Union[Scalar, ndarray] = func(arr, axis=axis, **kwargs)

        if isinstance(result, ndarray):
            arr = result
        else:
            arr = np.array([result])

        if arr.dtype.kind == 'U':
            arr = arr.astype('O')

        if arr.ndim == 1:
            if axis == 0:
                arr = arr[np.newaxis, :]
            else:
                arr = arr[:, np.newaxis]
        return arr

    def _stat_funcs(self, name: str, axis: str, **kwargs: Any) -> 'DataFrame':
        axis_int: int = utils.convert_axis_string(axis)
        data_dict: Dict[str, List[ndarray]] = defaultdict(list)

        good_dtypes: Set[str] = self._get_specific_stat_dtypes(name, axis_int)
        new_column_info: ColInfoT = {}
        change_kind: Dict[str, Tuple[str, int]] = {}
        new_num_cols: int = 0

        kind: str
        new_kind: str
        col: str
        loc: int
        add_loc: int
        order: int
        arr: ndarray
        hasnans: ndarray
        arr_new: ndarray
        cur_loc: int
        new_columns: ndarray
        func: Callable
        new_data: Dict[str, ndarray]

        if axis_int == 0:
            for kind, arr in self._data.items():
                if kind not in good_dtypes:
                    continue
                func = stat.funcs[kind][name]
                hasnans = self._hasnans_dtype(kind)
                kwargs.update({'hasnans': hasnans})
                arr_new = self._get_stat_func_result(func, arr, 0, kwargs)

                new_kind = arr_new.dtype.kind
                cur_loc = utils.get_num_cols(data_dict.get(new_kind, []))
                change_kind[kind] = (new_kind, cur_loc)
                data_dict[new_kind].append(arr_new)
                new_num_cols += arr_new.shape[1]

            new_columns = np.empty(new_num_cols, dtype='O')
            i: int = 0

            for col in self._columns:
                kind, loc, order = self._column_info[col].values
                if kind not in good_dtypes:
                    continue
                new_columns[i] = col
                new_kind, add_loc = change_kind[kind]
                new_column_info[col] = utils.Column(new_kind, loc + add_loc, i)
                i += 1

            new_data = utils.concat_stat_arrays(data_dict)
            return self._construct_from_new(new_data, new_column_info, new_columns)
        else:
            arrs: List[ndarray] = []
            result: ndarray
            if utils.is_column_stack_func(name):
                arr = self._values_raw(good_dtypes)
                kind = arr.dtype.kind
                if kind in 'fO':
                    hasnans = self._hasnans_dtype(kind)
                else:
                    hasnans = None
                kwargs.update({'hasnans': hasnans})
                func = stat.funcs[kind][name]
                result = self._get_stat_func_result(func, arr, 1, kwargs)
            else:
                for kind, arr in self._data.items():
                    if kind not in good_dtypes:
                        continue
                    func = stat.funcs[kind][name]
                    hasnans = self._hasnans_dtype(kind)
                    kwargs.update({'hasnans': hasnans})
                    arr_new = self._get_stat_func_result(func, arr, 1, kwargs)
                    arrs.append(arr_new)

                if len(arrs) == 1:
                    result = arrs[0]
                else:
                    func_across: Callable = stat.funcs_columns[name]
                    result = func_across(arrs)

            if utils.is_agg_func(name):
                new_columns = np.array([name], dtype='O')
            else:
                new_columns = [col for col in self._columns
                               if self._column_info[col].dtype in good_dtypes]

            new_kind = result.dtype.kind
            new_data = {new_kind: result}

            for i, col in enumerate(new_columns):
                new_column_info[col] = utils.Column(new_kind, i, i)
            return self._construct_from_new(new_data, new_column_info, new_columns)

    def sum(self, axis: str = 'rows') -> 'DataFrame':
        return self._stat_funcs('sum', axis)

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

    def abs(self, keep: bool = False) -> 'DataFrame':
        """
        Take the absolute value of each element.
        By default it will drop any non-numeric/bool columns
        Set `keep` to `True` to keep all columns

        Parameters
        ----------
        keep : bool - Set to `True` to keep all columns

        """
        if keep:
            df: 'DataFrame' = self
        else:
            df = self._get_numeric()

        new_data: Dict[str, ndarray] = {}
        dt: str
        arr: ndarray
        for dt, arr in df._data.items():
            if dt in 'ifbm':
                new_data[dt] = np.abs(arr)
            else:
                new_data[dt] = arr.copy()
        new_column_info: ColInfoT = df._copy_column_info()
        return df._construct_from_new(new_data, new_column_info, df._columns.copy())

    __abs__ = abs

    def _get_numeric(self) -> 'DataFrame':
        if ('i' not in self._data and 'f' not in self._data and
                'b' in self._data and 'm' not in self._data):
            raise TypeError('All columns must be either integer, float, or boolean')
        return self.select_dtypes(['number', 'bool', 'timedelta'])

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

    def count(self, axis: str = 'rows') -> 'DataFrame':
        return self._stat_funcs('count', axis)

    def _get_clip_df(self, value: Any, name: str, keep: bool) -> Tuple['DataFrame', str]:
        if value is None:
            raise ValueError('You must provide a value for either lower or upper')
        if utils.is_number(value):
            if self._has_numeric_or_bool():
                if keep:
                    df: 'DataFrame' = self
                else:
                    df = self.select_dtypes(['number', 'bool'])
                return df, 'number'
            else:
                raise TypeError(f'You provided a numeric value for {name} '
                                'but do not have any numeric columns')
        elif isinstance(value, str):
            if self._has_string():
                if keep:
                    df = self
                else:
                    df = self.select_dtypes('str')
                return df, 'str'
            else:
                raise TypeError(f'You provided a string value for {name} '
                                'but do not have any string columns')
        else:
            raise NotImplementedError('Data type incompatible')

    def clip(self, lower: Optional[ScalarT] = None, upper: Optional[ScalarT] = None,
             keep: bool = False) -> 'DataFrame':
        df: 'DataFrame'
        overall_dtype: 'str'
        if lower is None:
            df, overall_dtype = self._get_clip_df(upper, 'upper', keep)
        elif upper is None:
            df, overall_dtype = self._get_clip_df(lower, 'lower', keep)
        else:
            overall_dtype = utils.is_compatible_values(lower, upper)
            if lower > upper:
                raise ValueError('The upper value must be less than lower')

            if overall_dtype == 'number' and not keep:
                df = self.select_dtypes(['number', 'bool'])
            elif overall_dtype == 'str' and not keep:
                df = self.select_dtypes('str')
            else:
                df = self

        if overall_dtype == 'str':
            new_data: Dict[str, ndarray] = {}
            if lower is None:
                new_data['O'] = _math.clip_str_upper(df._data['O'], upper)
            elif upper is None:
                new_data['O'] = _math.clip_str_lower(df._data['O'], lower)
            else:
                new_data['O'] = _math.clip_str_both(df._data['O'], lower, upper)

            for kind, arr in self._data.items():
                if kind != 'O':
                    new_data[kind] = arr
        else:
            if utils.is_integer(lower) or utils.is_integer(upper):
                if 'b' in df._data:
                    as_type_dict = {col: 'int' for col, col_obj in df._column_info.items()
                                    if col_obj.dtype == 'b'}
                    df = df.astype(as_type_dict)
            if utils.is_float(lower) or utils.is_float(upper):
                if 'i' in df._data:
                    as_type_dict = {col: 'float' for col, col_obj in df._column_info.items()
                                    if col_obj.dtype == 'i'}
                    df = df.astype(as_type_dict)
                if 'b' in df._data:
                    as_type_dict = {col: 'float' for col, col_obj in df._column_info.items()
                                    if col_obj.dtype == 'f'}
                    df = df.astype(as_type_dict)
            new_data = {}
            for dtype, arr in df._data.items():
                if dtype in 'if':
                    new_data[dtype] = arr.clip(lower, upper)
                else:
                    new_data[dtype] = arr.copy(order='F')

        new_column_info: ColInfoT = df._copy_column_info()
        return df._construct_from_new(new_data, new_column_info, df._columns.copy())

    def cummax(self, axis: str = 'rows') -> 'DataFrame':
        return self._stat_funcs('cummax', axis)

    def cummin(self, axis: str = 'rows') -> 'DataFrame':
        return self._stat_funcs('cummin', axis)

    def cumsum(self, axis: str = 'rows') -> 'DataFrame':
        return self._stat_funcs('cumsum', axis)

    def cumprod(self, axis: str = 'rows') -> 'DataFrame':
        return self._stat_funcs('cumprod', axis)

    def mode(self, axis: str = 'rows', keep='low') -> 'DataFrame':
        if keep not in ('low', 'high', 'all'):
            raise ValueError('`keep` must be either "low", "high", or "all"')
        return self._stat_funcs('mode', axis, keep=keep)

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
        data_dict: Dict[str, List[ndarray]] = defaultdict(list)
        kind_shape: Dict[str, int] = OrderedDict()
        total_shape: int = 0

        kind: str
        arr: ndarray
        new_arr: ndarray
        for kind, arr in self._data.items():
            kind_shape[kind] = total_shape
            total_shape += arr.shape[1]
            if kind in 'bi':
                new_arr = np.empty_like(arr, dtype='bool')
                new_arr.fill(False)
                data_dict['b'].append(new_arr)
            elif kind == 'O':
                hasnans = self._hasnans_dtype('O')
                data_dict['b'].append(_math.isna_str(arr, hasnans))
            elif kind == 'f':
                hasnans = self._hasnans_dtype('f')
                data_dict['b'].append(_math.isna_float(arr, hasnans))
            elif kind in 'mM':
                # hasnans = self._hasnans_dtype('f')
                data_dict['b'].append(np.isnat(arr))

        new_column_info: ColInfoT = self._new_cd_from_kind_shape(kind_shape, 'b')
        new_data: Dict[str, ndarray] = utils.concat_stat_arrays(data_dict)
        return self._construct_from_new(new_data, new_column_info, self._columns.copy())

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
        data_dict: Dict[str, List[ndarray]] = defaultdict(list)
        new_column_info: ColInfoT = {}
        new_columns: List[str] = []

        if summary_type == 'numeric':

            data_dict['O'].append(df._columns.copy('F'))
            new_column_info['Column Name'] = utils.Column('O', 0, 0)
            new_columns.append('Column Name')

            dtypes = df._get_dtype_list()
            data_dict['O'].append(dtypes)
            new_column_info['Data Type'] = utils.Column('O', 1, 1)
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

            new_data: Dict[str, ndarray] = df._concat_arrays(data_dict)

        else:
            raise NotImplementedError('non-numeric summary not available yet')

        return self._construct_from_new(new_data, new_column_info, new_columns)

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
        new_column_info: ColInfoT = {'Column Name': utils.Column('O', 0, 0)}
        new_columns: ndarray = np.empty(x.shape[1] + 1, dtype='O')
        new_columns[0] = 'Column Name'

        i: int = 0
        col: str
        for col in self._columns:
            dtype, loc, order = self._column_info[col].values  # type: str, int, int
            if dtype not in 'ifb':
                continue
            new_column_info[col] = utils.Column('f', i, i + 1)
            new_columns[i + 1] = col
            i += 1
        new_data['O'] = np.asfortranarray(new_columns[1:])[:, np.newaxis]
        return self._construct_from_new(new_data, new_column_info, new_columns)

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
        new_column_info: ColInfoT = {'Column Name': utils.Column('O', 0, 0)}
        new_columns: ndarray = np.empty(x.shape[1] + 1, dtype='O')
        new_columns[0] = 'Column Name'

        i: int = 0
        col: str
        for col in self._columns:
            dtype, loc, order = self._column_info[col].values  # type: str, int, int
            if dtype not in 'ifb':
                continue
            new_column_info[col] = utils.Column('f', i, i + 1)
            new_columns[i + 1] = col
            i += 1
        new_data['O'] = np.asfortranarray(new_columns[1:])[:, np.newaxis]
        return self._construct_from_new(new_data, new_column_info, new_columns)

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
                if dtype == 'O':
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
                new_columns = [col]
                new_column_info = {col: utils.Column(kind, 0, 0)}
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
                    for col, loc in zip(new_columns, locs):
                        new_column_info[col] = utils.Column(dtype, loc, new_col_order[col])
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
                    if dtype == 'O':
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

        return self._construct_from_new(new_data, new_column_info, new_columns)

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
            if 'O' not in self._data:
                raise TypeError("You passed a `str` value to the `values` parameter. "
                                "You're DataFrame contains no str columns.")
            new_data = {}
            for dtype, arr in self._data.items():
                arr = arr.copy('F')
                if dtype == 'O':
                    for col in self._columns:
                        dtype2, loc, _ = self._column_info[col].values
                        if dtype2 == 'O':
                            col_arr = arr[:, loc]
                            na_arr: ndarray = _math.isna_str_1d(col_arr)
                            idx = np.where(na_arr)[0][:limit]
                            col_arr[idx] = values

                    new_data['O'] = arr
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
                dtype, loc, _ = self._column_info[col].values
                if dtype in 'fO':
                    dtype_locs[dtype].append((col, loc, val))

            for col, loc, new_val in dtype_locs['f']:
                if not isinstance(new_val, (int, float, np.number)):
                    raise TypeError(f'Column {col} has dtype float. Must set with a number')
            for col, loc, new_val in dtype_locs['O']:
                if not isinstance(new_val, str):
                    raise TypeError(f'Column {col} has dtype {dtype}. Must set with a str')

            arr_float: ndarray = self._data.get('f', []).copy('F')
            arr_str: ndarray = self._data.get('O', []).copy('F')
            for col, loc, new_val in dtype_locs['f']:
                if limit >= len(self):
                    arr_float[:, loc] = np.where(np.isnan(arr_float[:, loc]), new_val,
                                                 arr_float[:, loc])
                else:
                    idx = np.where(np.isnan(arr_float[:, loc]))[0][:limit]
                    arr_float[idx, loc] = new_val

            for col, loc, new_val in dtype_locs['O']:
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
                elif dtype == 'O':
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
                        elif dtype == 'O':
                            new_data['O'] = _math.ffill_str(arr, limit)
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
                        elif dtype == 'O':
                            new_data['O'] = _math.bfill_str(arr, limit)
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
        if dtype == 'O':
            if hasnans or hasnans is None:
                if asc:
                    nan_value = chr(10 ** 6)
                else:
                    nan_value = ''
                if col_arr.ndim == 1:
                    na_arr: ndarray = _math.isna_str_1d(col_arr)
                    arr_final: ndarray = np.where(na_arr, nan_value, col_arr)
                else:
                    hasnans = np.array([True] * col_arr.shape[1])
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
            dtype, loc, _ = self._column_info[col].values
            col_arr = self._data[dtype][:, loc]
            hasnans: ndarray = self._hasnans.get(col, True)
            asc = ascending[0]
            col_arr = self._replace_nans(dtype, col_arr, asc, hasnans)
            count_sort: bool = False
            if dtype == 'O':
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
                dtype, loc, _ = self._column_info[col].values
                col_arr = self._data[dtype][:, loc]
                hasnans = self._hasnans.get(col, True)
                col_arr = self._replace_nans(dtype, col_arr, asc, hasnans)

                if not asc:
                    if dtype == 'b':
                        col_arr = ~col_arr
                    elif dtype == 'O':
                        # TODO: how to avoid mapping to ints for mostly unique string columns?
                        d = _sr.sort_str_map(col_arr, asc)
                        col_arr = _sr.replace_str_int(col_arr, d)
                    elif dtype == 'M':
                        col_arr = (-(col_arr.view('int64') + 1)).astype('datetime64[ns]')
                    elif dtype == 'm':
                        col_arr = (-(col_arr.view('int64') + 1)).astype('timedelta64[ns]')
                    else:
                        col_arr = -col_arr
                elif dtype == 'O':
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

        for col in self._columns:
            dtype, loc, order = self._column_info[col].values
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
            for i, col in enumerate(self._columns):
                dtype, loc, order = self._column_info[col].values
                dtype_loc[dtype].append(loc)
                dtype_col[dtype].append(col)
                new_col_order[col] = i

            new_columns = self._columns.copy()
        else:
            col_set = set(subset)
            new_columns = np.empty(len(col_set), dtype='O')
            i = 0
            for col in self._columns:
                if col not in col_set:
                    continue
                new_columns[i] = col
                new_col_order[col] = i
                i += 1
                dtype, loc, order = self._column_info[col].values
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
                if dtype == 'O':
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

            if method == 'first' and dtype == 'O' and not ascending:
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
        data_dict: Dict[str, List[ndarray]] = defaultdict(list)
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
        dtype, loc, _ = self._column_info[col].values
        arr = self._data[dtype][:, loc]
        if dtype == 'O':
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

        new_columns = [col, 'count']

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

        dtype, loc, _ = self._column_info[column].values
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
                elif dtype == 'O':
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
            new_columns = self._check_column_validity(columns)

        elif isinstance(columns, dict):
            for col in columns:
                if col not in self._column_info:
                    raise ValueError(f'Column {col} is not a column')

            new_columns = [columns.get(col, col) for col in self._columns]
            new_columns = self._check_column_validity(new_columns)

        elif callable(columns):
            new_columns = [columns(col) for col in self._columns]
            new_columns = self._check_column_validity(new_columns)
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

        return self._construct_from_new(new_data, new_column_info, new_columns)

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
            dtype, loc, order = self._column_info[col].values
            dtype_drop_info[dtype].append(loc)

        new_column_info = {}
        new_columns = []
        order_sub = 0
        for col in self._columns:
            if col not in column_strings:
                dtype, loc, order = self._column_info[col].values
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

        return self._construct_from_new(new_data, new_column_info, new_columns)

    def _drop_just_rows(self, rows):
        if isinstance(rows, int):
            rows = [rows]
        elif isinstance(rows, ndarray):
            rows = utils.try_to_squeeze_array(rows)
        elif isinstance(rows, list):
            pass
        else:
            raise TypeError('Rows must either be an int, list/array of ints or None')

    def drop(self,
             rows: Union[int, List[int], ndarray, None] = None,
             columns: Union[int, List[IntStr], ndarray, None] = None):
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
            dtype, loc, _ = self._column_info[col].values
            cur_loc = len(data_dict[dtype])
            new_column_info[col] = utils.Column(dtype, cur_loc, i)
            data_dict[dtype].append(loc)

        new_data = {}
        for dtype, locs in data_dict.items():
            new_data[dtype] = self._data[dtype][np.ix_(new_rows, locs)]

        return self._construct_from_new(new_data, new_column_info, new_columns)

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
        dtype, loc, _ = self._column_info[column].values
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
            elif col_arr.dtype.kind == 'O':
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
                if dtype == 'O':
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
                if dtype == 'O':
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
        dtype, loc, _ = self._column_info[column].values
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
            data_dict: Dict[str, List[ndarray]] = defaultdict(list)
            new_columns = []
            new_column_info = {}
            for i, num in enumerate(new_idx):
                col = self._columns[num]
                dtype, loc, _ = self._column_info[col].values
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
            return self._construct_from_new(new_data, new_column_info, new_columns)

    def isin(self, values: Union[
        Scalar, List[Scalar], Dict[str, Union[Scalar, List[Scalar]]]]) -> 'DataFrame':
        if utils.is_scalar(values):
            values = [values]  # type: ignore

        def separate_value_types(vals: List[Scalar]) -> Tuple[
            List[Union[float, int, bool]], List[str]]:
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
                if dtype == 'O':
                    arrs.append(np.isin(arr, val_strings))
                elif dtype == 'M':
                    arrs.append(np.isin(arr, val_datetimes))
                elif dtype == 'm':
                    arrs.append(np.isin(arr, val_timedeltas))
                else:
                    arrs.append(np.isin(arr, val_numbers))

            new_column_info = {}
            for col in self._columns:
                dtype, loc, order = self._column_info[col].values
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
                dtype, loc, order = self._column_info[col].values
                col_arr = self._data[dtype][:, loc]
                val_numbers, val_strings, val_datetimes, val_timedeltas = separate_value_types(vals)
                if dtype == 'O':
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
                return None if dtype == 'O' else nan

            types: Any
            if dtype == 'O':
                good_dtypes = ['O', 'U']
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
                elif x.dtype.kind == 'O' and not isinstance(y, str):
                    raise TypeError('`x` and `y` arrays have incompatible dtypes. `x` is str '
                                    'and `y` is not')
                elif x.dtype.kind == 'b' and not isinstance(y, (bool, np.bool_)):
                    raise TypeError('`x` and `y` arrays have incompatible dtypes. `x` is bool '
                                    'and `y` is not')

            elif utils.is_scalar(x) and isinstance(y, ndarray):
                if y.dtype.kind in 'if' and not isinstance(x, (int, float, np.number)):
                    raise TypeError('`x` and `y` arrays have incompatible dtypes. `y` is numeric '
                                    'and `x` is not')
                elif y.dtype.kind == 'O' and not isinstance(x, str):
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

        data_dict: Dict[str, List[ndarray]] = defaultdict(list)
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

    def _ipython_key_completions_(self):
        return self._columns.tolist()
