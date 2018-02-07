from collections import defaultdict, OrderedDict
import warnings

import numpy as np
from numpy import nan, ndarray
from typing import (Union, Dict, List, Optional, Tuple, Sequence, Callable,
                    NoReturn, Set, Iterable, Any, TypeVar, Type, cast)

import dexplo.options as options
import dexplo.utils as utils
from dexplo._libs import (string_funcs as sf,
                          groupby as gb,
                          validate_arrays as va,
                          math as _math)
from dexplo import stat_funcs as stat

DataC = Union[Dict[str, Union[ndarray, List]], ndarray]

# can change to array of strings?
ColumnT = Optional[Union[List[str], ndarray]]
ColInfoT = Dict[str, utils.Column]

IntStr = TypeVar('IntStr', int, str)
IntNone = TypeVar('IntNone', int, None)
ListIntNone = List[IntNone]

ScalarT = TypeVar('ScalarT', int, float, str, bool)

ColSelection = Union[int, str, slice, List[IntStr]]
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
        self._check_column_validity(new_columns)
        new_columns = np.asarray(new_columns, dtype='O')

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
            self._check_column_validity(columns)
        else:
            self._check_column_validity(columns)
            if set(columns) != set(data.keys()):
                raise ValueError("Column names don't match dictionary keys")

        self._columns: ColumnT = np.asarray(columns, dtype='O')

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
            self._check_column_validity(columns)
            if len(columns) != num_cols:
                raise ValueError(f'Number of column names {len(columns)} does not equal '
                                 f'number of columns of data array {num_cols}')
            self._columns: ColumnT = np.asarray(columns, dtype='O')

    def _check_column_validity(self, cols: ColumnT) -> None:
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
        data_dict: Dict[str, List[ndarray]] = {'f': [], 'i': [], 'b': [], 'O': []}
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
        if kind == 'S':
            data = data.astype('U').astype('O')
        elif kind == 'U':
            data = data.astype('O')

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
            arr: ndarray = va.maybe_convert_object_array(data[:, i], col)
            kind: str = arr.dtype.kind
            loc: int = len(data_dict[kind])
            data_dict[kind].append(arr)
            self._column_info[col] = utils.Column(kind, loc, i)

        self._data = self._concat_arrays(data_dict)

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
            order = [col_obj.loc for col, col_obj in self._column_info.items()]
            return self._data[kind].copy('F')[:, order]

        if 'b' in self._data or 'O' in self._data:
            arr_dtype: str = 'O'
        else:
            arr_dtype = 'float64'

        v: ndarray = np.empty(self.shape, dtype=arr_dtype, order='F')

        for col, col_obj in self._column_info.items():
            dtype, loc, order2 = col_obj.values  # type: str, int, int
            v[:, order2] = self._data[dtype][:, loc]
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

    def _values_number_drop(self, columns: list, dtype_loc: Dict[str, List[int]],
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
        # if len(self._data) == 1:
        #     kind: str = list(self._data.keys())[0]
        #     return self._data[kind]

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
                data = [column] + self._get_column_values(column)[idx].tolist()
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

            data_list.append(data)

        return data_list, long_len, decimal_len, idx

    # def __repr__(self) -> str:
    #     data_list: List[List[str]]
    #     decimal_len: List[int]
    #     idx: List[int]
    #     data_list, long_len, decimal_len, idx = self._build_repr()
    #
    #     return_string: str = ''
    #     for i in range(len(idx) + 1):
    #         for d, fl, dl in zip(data_list, long_len, decimal_len):
    #             print('d is', d)
    #             if isinstance(d[i], (float, np.floating)) and np.isnan(d[i]):
    #                 d[i] = 'NaN'
    #             if isinstance(d[i], bool):
    #                 d[i] = str(d[i])
    #             if isinstance(d[i], str):
    #                 cur_str = d[i]
    #                 if len(cur_str) > options.max_colwidth:
    #                     cur_str = cur_str[:options.max_colwidth - 3] + "..."
    #                 return_string += f'{cur_str: >{fl}}  '
    #             else:
    #                 print(i, d[i])
    #                 return_string += f'{d[i]: >{fl}.{dl}f}  '
    #         return_string += '\n'
    #         if i == options.max_rows // 2 and len(self) > options.max_rows:
    #             for j, fl in enumerate(long_len):
    #                 return_string += f'{"...": >{str(fl)}}'
    #             return_string += '\n'
    #     return return_string

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
                                     f'{len(row_array)} != {len(self)}')
            elif row_array.dtype.kind != 'i':
                raise TypeError('Row selection array data type must be either integer or boolean')
        elif isinstance(rs, DataFrame):
            if rs.shape[1] != 1:
                raise ValueError('Boolean selection only works with single-column DataFames')
            row_array = rs.values.squeeze()
            if row_array.dtype.kind != 'b':
                raise TypeError('All values for row selection must be boolean')
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
                arr = np.atleast_2d(self._data[dtype][ix]).copy()
            else:
                arr = np.atleast_2d(self._data[dtype][rs, pos]).copy()
            if arr.dtype.kind == 'U':
                arr = arr.astype('O')
            elif arr.dtype.kind == 'S':
                arr = arr.astype('U').astype('O')
            new_data[dtype] = np.asfortranarray(arr)

        return self._construct_from_new(new_data, new_column_info, new_columns)

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
        for col, col_obj in self._column_info.items():
            dtype, loc, order = col_obj.values  # type: str, int, int
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
        df_new: 'DataFrame' = cls.__new__(cls)
        df_new._column_info = column_info
        df_new._data = data
        df_new._columns = np.asarray(columns, dtype='O')
        df_new._hasnans = {}
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
                # will have to do custom cython function here for add,
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

    def _op(self, other: Any, op_string: str) -> 'DataFrame':
        if isinstance(other, (int, float, bool)):
            if self._is_numeric_or_bool() or op_string in ['__mul__', '__rmul__']:
                dd, ncd = self._do_eval(op_string, other)
            else:
                raise TypeError('You have a mix of numeric and string data '
                                'types. Operation is ambiguous.')

        elif isinstance(other, str):
            if self._is_string():
                if op_string in ['__add__', '__radd__', '__gt__', '__ge__',
                                 '__lt__', '__le__', '__ne__', '__eq__']:
                    dd, ncd = self._do_eval(op_string, other)
            else:
                raise TypeError('All columns in DataFrame must be string if '
                                'operating with a string')

        elif isinstance(other, DataFrame):
            raise NotImplementedError('Operating with two DataFrames will be '
                                      'developed in the future')
        else:
            raise TypeError('other must be int, float, or DataFrame')

        new_data: Dict[str, ndarray] = {}
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
                new_data[dt] = -arr.copy()
        else:
            raise TypeError('Only works for all numeric columns')
        new_column_info: ColInfoT = {col: utils.Column(*col_obj.values)
                                     for col, col_obj in self._column_info.items()}
        return self._construct_from_new(new_data, new_column_info,
                                        self._columns.copy())

    def __bool__(self) -> NoReturn:
        raise ValueError(': The truth value of an array with more than one element is ambiguous. '
                         'Use a.any() or a.all()')

    def __getattr__(self, name: str) -> NoReturn:
        dtypes: Dict[str, str] = {'dt': 'datetime',
                                  'str': 'object',
                                  'cat': 'Categorical'}
        if name in dtypes:
            raise AttributeError(f'The {name} accessor is only available for '
                                 'single-column DataFrames with data type '
                                 f'{dtypes[name]}')
        else:
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
            nulls: ndarray = np.isnan(col_data)
            col_data = col_data.astype('U').astype('O')
            col_data[nulls] = None
        else:
            col_data = col_data.astype(numpy_dtype)
        self._remove_column(column)
        self._write_new_column_data(column, new_kind, col_data, order)

    def _remove_column(self, column: str) -> None:
        """
        Removes column from _colum_dtype, and _data
        Keeps column name in _columns
        """
        dtype, loc, order = self._column_info.pop(column).values  # type: str, int, int
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
            self._data[new_kind] = np.column_stack((self._data[new_kind], data))
        else:
            if data.ndim == 1:
                data = data[:, np.newaxis]
            self._data[new_kind] = data

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
                va.validate_strings_in_object_array(value)
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
                arr = arr.astype(new_dtype)
                if arr.dtype.kind == 'U':
                    arr = arr.astype('O')
                new_kind = arr.dtype.kind
                data_dict[new_kind].append(arr)

            new_column_info: ColInfoT = self._new_cd_from_kind_shape(kind_shape, new_kind)
            new_data: Dict[str, ndarray] = utils.concat_stat_arrays(data_dict)
            return self._construct_from_new(new_data, new_column_info,
                                            self._columns.copy())

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
                if kind == 'f':
                    self._hasnans['f'] = np.isnan(arr).any(0)
                elif kind == 'O':
                    self._hasnans['O'] = va.isnan_object(arr)
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
                                f'The `{name}` only works with numeric columns.')
            return {'O'}
        if axis == 0:
            if name in ['std', 'var', 'mean', 'median', 'quantile']:
                return {'i', 'f', 'b'}
        elif axis == 1:
            if name in ['std', 'var', 'mean', 'median', 'quantile', 'sum', 'max', 'min',
                        'cumsum', 'cummax', 'cummin', 'argmin', 'argmax']:
                return {'i', 'f', 'b'}
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
            if dt in 'ifb':
                new_data[dt] = np.abs(arr)
            else:
                new_data[dt] = arr.copy()
        new_column_info: ColInfoT = df._copy_column_info()
        return df._construct_from_new(new_data, new_column_info, df._columns.copy())

    __abs__ = abs

    def _get_numeric(self) -> 'DataFrame':
        if not self._has_numeric_or_bool():
            raise TypeError('All columns must be either integer, float, or boolean')
        return self.select_dtypes(['number', 'bool'])

    def any(self, axis: str = 'rows') -> 'DataFrame':
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

    def _hasnans_dtype(self, kind: str) -> ndarray:
        hasnans: ndarray = np.ones(self._data[kind].shape[1], dtype='bool')
        return self._hasnans.get(kind, hasnans)

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

    def _null_pct(self):
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
            df = self.select_dtypes(['str', 'bool'])
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

    def cov(self):
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

    def corr(self):
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

    def unique(self, col: str) -> ndarray:
        """
        Finds the unique elements of a single column in the order that they appeared

        Parameters
        ----------
        col : Name of column as a string

        Returns
        -------
        A one dimensional NumPy array
        """
        self._validate_column_name(col)
        kind, loc, _ = self._column_info[col].values  # type: str, int, int
        arr: ndarray = self._data[kind][:, loc]

        if kind == 'i':
            return _math.unique_int(arr)
        elif kind == 'f':
            return _math.unique_float(arr)
        elif kind == 'b':
            return _math.unique_bool(arr)
        return _math.unique_str(arr)

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

    def value_counts(self, col):
        self._validate_column_name(col)
        arr = self._get_col_array(col)

        group_labels, group_position = gb.get_group_assignment(arr)

        size = gb.size(group_labels, d)

        new_data = {'O': arr[group_position], 'b': None, 'f': None, 'i': size[:, np.newaxis]}
        new_columns = ['Column Values', 'Counts']
        # add _column_info
        return self._construct_from_new(new_data, new_columns)

    def groupby(self, columns):
        if isinstance(columns, list):
            col_set = set()
            for col in columns:
                self._validate_column_name(col)
                if col in col_set:
                    raise ValueError('`{col}` has already been selected as a grouping column')
                col_set.add(col)
        elif isinstance(columns, str):
            self._validate_column_name(columns)
            columns = [columns]
        else:
            raise ValueError('Must pass in grouping column(s) as a string or list of strings')
        return Grouper(self, columns)


class Grouper(object):

    def __init__(self, df: DataFrame, columns: Union[str, List[str]]) -> None:
        self._df: DataFrame = df
        self._group_labels, self._group_position = self._create_groups(columns)
        self._group_columns = columns

    def _create_groups(self, columns: List[str]) -> Tuple:
        self._group_dtype_loc: Dict[str, List[int]] = defaultdict(list)
        self._column_info = {}
        for i, col in enumerate(columns):
            dtype, loc, _ = self._df._column_info[col].values  # type: str, int, int
            cur_loc = len(self._group_dtype_loc[dtype])
            self._group_dtype_loc[dtype].append(loc)
            self._column_info[col] = utils.Column(dtype, cur_loc, i)

        if len(columns) == 1:
            # since there is just one column, dtype is from the for-loop
            final_arr = self._df._data[dtype][:, loc].squeeze()
            if dtype == 'O':
                return gb.get_group_assignment_str_1d(final_arr)
            if dtype == 'i':
                return gb.get_group_assignment_int_1d(final_arr)
            if dtype == 'f':
                return gb.get_group_assignment_float_1d(final_arr)
            if dtype == 'b':
                return gb.get_group_assignment_bool_1d(final_arr)
        else:
            arrs = []

            total_locs = 0
            start = 0
            end = 0
            for i, (dtype, locs) in enumerate(self._group_dtype_loc.items()):
                # in case there is a mix of floats and objects
                if dtype == 'f':
                    start = total_locs
                    end = start + len(locs)
                    float_idx = i
                arrs.append(self._df._data[dtype][:, locs])
                total_locs += len(locs)

            final_arr = np.column_stack(arrs)
            final_dtype = final_arr.dtype.kind

            if final_dtype == 'O' and end != 0:
                arr_nan = np.isnan(arrs[float_idx])
                final_arr[:, start:end][arr_nan] = None

            if final_dtype == 'O':
                return gb.get_group_assignment_str_2d(final_arr)
            if final_dtype == 'i':
                return gb.get_group_assignment_int_2d(final_arr)
            if final_dtype == 'f':
                return gb.get_group_assignment_float_2d(final_arr)
            if final_dtype == 'b':
                return gb.get_group_assignment_bool_2d(final_arr)
        if len(self._group_position) == self._df.shape[0]:
            warnings.warn("Each group contains exactly one row of data. "
                          "Are you sure you are grouping correctly?")

    def _get_group_col_data(self):
        data_dict = defaultdict(list)
        for dtype, locs in self._group_dtype_loc.items():
            ix = np.ix_(self._group_position, locs)
            arr = self._df._data[dtype][ix]
            if arr.ndim == 1:
                arr = arr[:, np.newaxis]
            data_dict[dtype].append(arr)
        return data_dict

    def _get_group_col_data_all(self):
        data_dict = defaultdict(list)
        for dtype, locs in self._group_dtype_loc.items():
            arr = self._df._data[dtype][:, locs]
            if arr.ndim == 1:
                arr = arr[:, np.newaxis]
            data_dict[dtype].append(arr)
        return data_dict

    def _get_agg_name(self, name):
        i = 1
        while name in self._group_columns:
            name = name + str(i)
        return name

    def __repr__(self):
        return ("This is a groupby object. Here is some info on it:\n"
                f"Grouping Columns: {self._group_columns}\n"
                f"Number of Groups: {len(self._group_position)}")

    def __len__(self):
        return len(self._group_position)

    def _get_new_column_info(self):
        new_column_info = {}
        for col, col_obj in self._column_info.items():
            new_column_info[col] = utils.Column(*col_obj.values)
        return new_column_info

    @property
    def ngroups(self):
        return len(self._group_position)

    def _group_agg(self, name, ignore_str=True, add_positions=False, **kwargs):
        labels = self._group_labels
        size = len(self._group_position)
        data_dict = self._get_group_col_data()

        old_dtype_col = defaultdict(list)
        for col, col_obj in self._df._column_info.items():
            if col not in self._group_columns:
                old_dtype_col[col_obj.dtype].append(col)

        new_column_info = self._get_new_column_info()

        for dtype, data in self._df._data.items():
            if ignore_str and dtype == 'O':
                continue
            # number of grouped columns
            group_locs: list = self._group_dtype_loc.get(dtype, [])
            if len(group_locs) != data.shape[1]:
                func_name = name + '_' + utils.convert_kind_to_dtype(dtype)
                func = getattr(gb, func_name)
                if add_positions:
                    arr = func(labels, size, data, group_locs, self._group_position, **kwargs)
                else:
                    arr = func(labels, size, data, group_locs, **kwargs)
            else:
                continue

            new_kind = arr.dtype.kind
            cur_loc = utils.get_num_cols(data_dict.get(new_kind, []))
            data_dict[new_kind].append(arr)

            for col in old_dtype_col[dtype]:
                count_less = 0
                old_kind, old_loc, old_order = self._df._column_info[col].values
                for k in self._group_dtype_loc.get(dtype, []):
                    count_less += old_loc > k

                new_column_info[col] = utils.Column(new_kind, cur_loc + old_loc - count_less, 0)

        new_columns = self._group_columns.copy()
        i = len(self._group_columns)
        j = 0
        for col in self._df._columns:
            if col not in new_column_info:
                continue
            if col in self._group_columns:
                new_column_info[col].order = j
                j += 1
                continue
            # new_columns[i] = col
            new_columns.append(col)
            new_column_info[col].order = i
            i += 1

        new_data = utils.concat_stat_arrays(data_dict)

        return DataFrame._construct_from_new(new_data, new_column_info, new_columns)

    def size(self):
        name = self._get_agg_name('size')
        new_columns = np.array(self._group_columns + [name], dtype='O')
        size = gb.size(self._group_labels, len(self._group_position))[:, np.newaxis]
        data_dict = self._get_group_col_data()
        data_dict['i'].append(size)
        new_data = utils.concat_stat_arrays(data_dict)
        new_column_info = self._get_new_column_info()
        new_column_info[name] = utils.Column('i', new_data['i'].shape[1] - 1,
                                             len(new_columns) - 1)
        return DataFrame._construct_from_new(new_data, new_column_info, new_columns)

    def count(self):
        return self._group_agg('count', ignore_str=False)

    def cumcount(self):
        name = self._get_agg_name('cumcount')
        new_columns = np.array(self._group_columns + [name], dtype='O')
        cumcount = gb.cumcount(self._group_labels, len(self._group_position))[:, np.newaxis]
        data_dict = self._get_group_col_data_all()
        data_dict['i'].append(cumcount)
        new_data = utils.concat_stat_arrays(data_dict)
        new_column_info[name] = utils.Column('i', new_data['i'].shape[1] - 1,
                                             len(new_columns) - 1)
        return DataFrame._construct_from_new(new_data, new_column_info, new_columns)

    def sum(self):
        return self._group_agg('sum')

    def mean(self):
        return self._group_agg('mean')

    def max(self):
        return self._group_agg('max', False)

    def min(self):
        return self._group_agg('min', False)

    def first(self):
        new_columns = self._group_columns.copy()
        for col in self._df._columns:
            if col in self._group_columns:
                continue
            new_columns.append(col)
        return self._df[self._group_position, new_columns]

    def last(self):
        return self._group_agg('last', False)

    def var(self, ddof=1):
        return self._group_agg('var', add_positions=True, ddof=1)

    def cov(self):
        return self._cov_corr('cov')

    def corr(self):
        return self._cov_corr('corr')

    def _cov_corr(self, name):
        calc_columns = []
        calc_dtype_loc = []
        np_dtype = 'int64'
        for col in self._df._columns:
            if col in self._group_columns:
                continue
            dtype, loc, order = self._df._column_info[col].values
            if dtype in 'fib':
                if dtype == 'f':
                    np_dtype = 'float64'
                calc_columns.append(col)
                calc_dtype_loc.append((dtype, loc))

        data = self._df._values_number_drop(calc_columns, calc_dtype_loc, np_dtype)
        dtype_word = utils.convert_kind_to_dtype(data.dtype.kind)
        func = getattr(gb, name + '_' + dtype_word)
        result = func(self._group_labels, len(self), data, [])

        data_dict = self._get_group_col_data()
        data_dict_final = defaultdict(list)
        for dtype, arrs in data_dict.items():
            data_dict_final[dtype] = [np.repeat(arrs[0], len(calc_columns), axis=0)]

        new_column_info = self._get_new_column_info()
        num_group_cols = len(self._group_columns)
        new_columns = self._group_columns.copy()

        cur_obj_loc = utils.get_num_cols(data_dict_final.get('O', []))
        column_name_array = np.tile(calc_columns, len(self))[:, np.newaxis]
        data_dict_final['O'].append(column_name_array)
        new_columns.append('Column Name')
        new_column_info['Column Name'] = utils.Column('O', cur_obj_loc, num_group_cols)

        cur_loc = utils.get_num_cols(data_dict_final.get('f', []))

        for i, col in enumerate(calc_columns):
            new_column_info[col] = utils.Column('f', i + cur_loc, i + num_group_cols + 1)
            new_columns.append(col)

        data_dict_final['f'].append(result)
        new_data = utils.concat_stat_arrays(data_dict_final)

        return DataFrame._construct_from_new(new_data, new_column_info, new_columns)

    def any(self):
        return self._group_agg('any', False)

    def all(self):
        return self._group_agg('all', False)

    def median(self):
        return self._group_agg('median')

    def nunique(self):
        return self._group_agg('nunique', False)

    def head(self, n=5):
        row_idx = gb.head(self._group_labels, len(self), n=n)
        return self._df[row_idx, :]

    def tail(self, n=5):
        row_idx = gb.tail(self._group_labels, len(self), n=n)
        return self._df[row_idx, :]

class StringClass(object):

    def __init__(self, df):
        self.df = df

    def capitalize(self):
        arr = sf.capitalize(self.df._data['O'][:, 0])
        return self._create_df(arr, 'O')

    def capitalize2(self):
        arr = sf.capitalize2(self.df._data['O'][:, 0])
        return self._create_df(arr, 'O')

    def capitalize3(self, arr):
        return sf.capitalize3(arr)

    def cat(self):
        return ''.join(self.df._data['O'][:, 0].tolist())

    def center(self, width, fill_character=' '):
        arr = sf.center(self.df._data['O'][:, 0], width, fill_character)
        return self._create_df(arr, 'O')

    def contains(self, pat, case=True, flags=0, na=nan, regex=True):
        arr = sf.contains(self.df._data['O'][:, 0], case, flags, na, regex)
        return self._create_df(arr, 'b')

    def count(self, pattern):
        arr = sf.count(self.df._data['O'][:, 0], pattern)
        return self._create_df(arr, 'i')

    def _create_df(self, arr, dtype):
        arr = arr[:, np.newaxis]
        new_columns = self.df.columns
        new_data: Block = {'O': None, 'b': None, 'f': None, 'i': None}
        new_data[dtype] = arr
        # add _column_info
        return self.df._construct_from_new(new_data, new_columns)
