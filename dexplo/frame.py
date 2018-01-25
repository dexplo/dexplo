import decimal
from functools import partial
import numpy as np
from typing import (Union, Dict, List, Optional, Tuple,
                    NoReturn, Any, Set, Sequence)
import bottleneck as bn
import pandas_lite.utils as utils
import pandas_lite.options as options
import copy
from pandas_lite._libs import (string_funcs as sf,
                               groupby as gb,
                               validate_arrays as va,
                               math as _math,
                               stat_funcs as stat)

from numpy import nan, ndarray
from collections import defaultdict, OrderedDict


DataC = Union[Dict[str, Union[ndarray, List]], ndarray]
ColC = Dict[str, ndarray]
DtypeCol = Dict[str, List[str]]

# can change to array of strings?
ColumnT = Optional[Union[List[str], ndarray]]

ColumnSelection = Union[int, str, slice, List[Union[str, int]]]
RowSelection = Union[int, slice, List[int]]

_NUMERIC_KINDS: Set[str] = set('bif')
_ALL_DTYPES: Set[str] = set('bifOM')


class DataFrame(object):

    def __init__(self, data: DataC, columns: ColumnT=None) -> None:

        self._columns: ndarray
        self._data: Dict[str, ndarray] = {}
        self._column_dtype: Dict[str, utils.Column] = {}
        self._hasnans = {}

        if isinstance(data, dict):
            self._initialize_columns_from_dict(columns, data)
            self._initialize_data_from_dict(data)

        elif isinstance(data, ndarray):
            num_cols = self._validate_data_from_array(data)
            self._initialize_columns_from_array(columns, num_cols)

            if data.dtype.kind == 'O':
                self._initialize_from_object_array(data)
            else:
                self._initialize_data_from_array(data)

        else:
            raise TypeError('data parameter must be either a dict '
                            'of arrays or an array')

        # self._add_accessors()

    @property
    def columns(self) -> List[str]:
        return self._columns.tolist()

    # Only gets called after construction, when renaming columns
    @columns.setter
    def columns(self, new_columns: ColumnT) -> None:
        self._check_column_validity(new_columns)
        new_columns = np.asarray(new_columns, dtype='O')

        len_new, len_old = len(new_columns), len(self._columns)
        if len_new != len_old:
            raise ValueError(f'''There are {len_old} columns in the DataFrame.
                              You provided {len_new}''')

        new_data = {}
        new_column_dtype = {}
        for old_col, new_col in zip(self._columns, new_columns):
            new_data[new_col] = self._data[old_col]
            new_column_dtype[new_col] = self._column_dtype[old_col]

        self._data = new_data
        self._column_dtype = new_column_dtype
        self._columns = new_columns

    def _initialize_columns_from_dict(self, columns: ColumnT,
                                      data: DataC) -> None:
        if columns is None:
            columns = np.array(list(data.keys()))
            self._check_column_validity(columns)
        else:
            self._check_column_validity(columns)
            if set(columns) != set(data.keys()):
                raise ValueError("Column names don't match dictionary keys")

        self._columns = np.asarray(columns, dtype='O')

    def _initialize_columns_from_array(self, columns: ColumnT,
                                       num_cols: int) -> None:
        if columns is None:
            col_list = ['a' + str(i) for i in range(num_cols)]
            self._columns = np.array(col_list, dtype='O')
        else:
            self._check_column_validity(columns)
            if len(columns) != num_cols:
                raise ValueError(f'Number of column names {len(columns)}'
                                 'does not equal number of columns '
                                 f'of data array {num_cols}')
            self._columns = np.asarray(columns, dtype='O')

    def _check_column_validity(self, cols: ColumnT) -> None:
        if not isinstance(cols, (list, ndarray)):
            raise TypeError('Columns must be a list or an array')
        if isinstance(cols, ndarray):
            cols = utils.try_to_squeeze_array(cols)

        col_set = set()
        for i, col in enumerate(cols):
            if not isinstance(col, str):
                raise TypeError('Column names must be a string')
            if col in col_set:
                raise ValueError(f'Column name {col} is duplicated. '
                                 'Column names must '
                                 'be unique')
            col_set.add(col)

    def _initialize_data_from_dict(self, data: DataC) -> None:

        data_dict: ColC = {'f': [], 'i': [], 'b': [], 'O': []}
        data_kind = {}
        for i, (col, values) in enumerate(data.items()):
            if not isinstance(values, (list, ndarray)):
                raise TypeError('Values of dictionary must be an '
                                'array or a list')
            if isinstance(values, list):
                arr = utils.convert_list_to_single_arr(values, col)
            else:
                arr = values
            arr = utils.maybe_convert_1d_array(arr, col)
            kind = arr.dtype.kind
            loc = len(data_dict.get(kind, []))
            data_dict[kind].append(arr)
            self._column_dtype[col] = utils.Column(kind, loc, i)

            if i == 0:
                first_len = len(arr)
            elif len(arr) != first_len:
                raise ValueError('All columns must be the same length')
        self._concat_arrays(data_dict)

    def _validate_data_from_array(self, data: ndarray) -> int:
        if data.dtype.kind not in 'bifSUO':
            raise TypeError('Array must be of type boolean, integer, '
                            'float, string, or unicode')
        if data.ndim == 1:
            return 1
        elif data.ndim == 2:
            return data.shape[1]
        else:
            raise ValueError('Array must be either one or two dimensions')

    def _initialize_data_from_array(self, data: ndarray) -> None:
        kind = data.dtype.kind
        if kind == 'S':
            data = data.astype('U').astype('O')
        elif kind == 'U':
            data = data.astype('O')

        if data.ndim == 1:
            data = data[:, np.newaxis]

        kind = data.dtype.kind
        self._data[kind] = np.asfortranarray(data)
        self._column_dtype = {col: utils.Column(kind, i, i)
                              for i, col in enumerate(self._columns)}

    def _initialize_from_object_array(self, data: ndarray) -> None:
        """
        Used to convert individial columns in object array
        """
        data_dict: ColC = {'f': [], 'i': [], 'b': [], 'O': []}
        for i, col in enumerate(self._columns):
            arr = va.maybe_convert_object_array(data[:, i], col)
            kind = arr.dtype.kind
            loc = utils.get_arr_length(data_dict.get(kind, []))
            data_dict[kind].append(arr)
            self._column_dtype[col] = utils.Column(kind, loc, i)
        # use utils.concat_stat_arrays?
        self._concat_arrays(data_dict)

    def _concat_arrays(self, data_dict):
        for dtype, arrs in data_dict.items():
            if arrs:
                if len(arrs) == 1:
                    self._data[dtype] = arrs[0][:, np.newaxis]
                else:
                    arrs = np.column_stack(arrs)
                    self._data[dtype] = np.asfortranarray(arrs)

    def _add_accessors(self):
        self.str = StringClass(self)

    @property
    def values(self) -> ndarray:
        if len(self._data) == 1:
            kind = list(self._data.keys())[0]
            return self._data[kind].copy('F')

        if 'b' in self._data or 'O' in self._data:
            dtype = 'O'
        else:
            dtype = 'float64'

        v = np.empty(self.shape, dtype=dtype, order='F')

        for col, col_obj in self._column_dtype.items():
            dtype, loc, order = col_obj.values
            v[:, order] = self._data[dtype][:, loc]
        return v

    def _values_number(self) -> ndarray:
        if len(self._data) == 1:
            kind = list(self._data.keys())[0]
            return self._data[kind]

        v = np.empty(self.shape, dtype='float64', order='F')

        for col, col_obj in self._column_dtype.items():
            dtype, loc, order = col_obj.values
            v[:, order] = self._data[dtype][:, loc]
        return v

    def _values_raw(self) -> ndarray:
        if len(self._data) == 1:
            kind = list(self._data.keys())[0]
            return self._data[kind].copy('F')

        if 'O' in self._data:
            dtype = 'O'
        elif 'f' in self._data:
            dtype = 'float64'
        else:
            dtype = 'int64'

        v = np.empty(self.shape, dtype=dtype, order='F')

        for col, col_obj in self._column_dtype.items():
            dtype, loc, order = col_obj.values
            v[:, order] = self._data[dtype][:, loc]
        return v

    def _get_column_values(self, col):
        dtype, loc, _ = self._column_dtype[col].values
        return self._data[dtype][:, loc]

    def _build_repr(self) -> Tuple:
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

            if self._column_dtype[column].dtype == 'O':
                cur_len = max([len(str(x)) for x in data])
                cur_len = min(cur_len, options.max_colwidth)
                long_len.append(cur_len)
                decimal_len.append(0)
            elif self._column_dtype[column].dtype == 'f':
                dec_len = [utils._get_decimal_len(x) for x in data[1:]]
                whole_len = [utils._get_whole_len(x) for x in data[1:]]

                dec_len = np.array(dec_len).clip(0, 6)
                whole_len = np.array(whole_len)
                lengths = [len(column), (dec_len + whole_len).max() + 1]

                max_decimal = dec_len.max()
                long_len.append(max(lengths))
                decimal_len.append(min(max_decimal, 6))
            elif self._column_dtype[column].dtype == 'i':
                lengths = [len(column)] + [len(str(x)) for x in data[1:]]
                long_len.append(max(lengths))
                decimal_len.append(0)
            elif self._column_dtype[column].dtype == 'b':
                long_len.append(max(len(column), 5))
                decimal_len.append(0)

            data_list.append(data)

        return data_list, long_len, decimal_len, idx

    def __repr__(self) -> str:
        data_list, long_len, decimal_len, idx = self._build_repr()

        string: str = ''
        for i in range(len(idx) + 1):
            for d, fl, dl in zip(data_list, long_len, decimal_len):
                if str(d[i]) == 'nan':
                    d[i] = 'NaN'
                if isinstance(d[i], bool):
                    d[i] = str(d[i])
                if isinstance(d[i], str):
                    cur_str = d[i]
                    if len(cur_str) > options.max_colwidth:
                        cur_str = cur_str[:options.max_colwidth - 3] + "..."
                    string += f'{cur_str: >{fl}}  '
                else:
                    string += f'{d[i]: >{fl}.{dl}f}  '
            string += '\n'
            if i == options.max_rows // 2 and len(self) > options.max_rows:
                for j, fl in enumerate(long_len):
                    string += f'{"...": >{str(fl)}}'
                string += '\n'
        return string

    def _repr_html_(self) -> str:
        data_list, long_len, decimal_len, idx = self._build_repr()

        string: str = '<table>'
        for i in range(len(idx) + 1):
            if i == 0:
                string += '<thead>'
            elif i == 1:
                string += '<tbody>'
            string += '<tr>'
            for j, (d, fl, dl) in enumerate(zip(data_list, long_len,
                                                decimal_len)):
                if str(d[i]) == 'nan':
                    d[i] = 'NaN'
                ts = '<th>' if j * i == 0 else '<td>'
                te = '</th>' if j * i == 0 else '</td>'
                if isinstance(d[i], bool):
                    d[i] = str(d[i])
                if isinstance(d[i], str):
                    cur_str = d[i]
                    if len(cur_str) > options.max_colwidth:
                        cur_str = cur_str[:options.max_colwidth - 3] + "..."
                    string += f'{ts}{cur_str: >{fl}}{te}'
                else:
                    string += f'{ts}{d[i]: >{fl}.{dl}f}{te}'
            if i == options.max_rows // 2 and len(self) > options.max_rows:
                string += '<tr>'
                for j, fl in enumerate(long_len):
                    ts = '<th>' if j == 0 else '<td>'
                    te = '</th>' if j == 0 else '</td>'
                    string += f'{ts}{"...": >{str(fl)}}{te}'
                string += '</tr>'
            string += '</tr>'
            if i == 0:
                string += '</thead>'
        return string + '</tbody></table>'

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
            return self._column_dtype[col].order
        except KeyError:
            raise KeyError(f'{col} is not in the columns')

    def _get_col_name_from_int(self, iloc: int) -> str:
        try:
            return self._columns[iloc]
        except IndexError:
            raise IndexError(f'Index {iloc} is out of bounds for '
                             'the columns')

    def _get_list_of_cols_from_selection(self, cs):
        new_cols: List[str] = []
        for i, col in enumerate(cs):
            if isinstance(col, bool):
                if col:
                    new_cols.append(self._get_col_name_from_int(i))
            elif isinstance(col, int):
                new_cols.append(self._get_col_name_from_int(col))
            elif col not in self._column_dtype:
                raise ValueError(f'{col} is not in the columns')
            else:
                new_cols.append(col)

        utils.check_duplicate_list(new_cols)
        return new_cols

    def _convert_col_selection(self, cs: ColumnSelection) -> List[str]:
        if isinstance(cs, str):
            self._validate_column_name(cs)
            cs = [cs]
        elif isinstance(cs, int):
            cs = [self._get_col_name_from_int(cs)]
        elif isinstance(cs, slice):
            sss: List[Optional[int]] = []
            for s in ['start', 'stop', 'step']:
                value: Optional[Union[str, int]] = getattr(cs, s)
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
            cs = self._columns[slice(*sss)]
        elif isinstance(cs, list):
            cs = self._get_list_of_cols_from_selection(cs)
        elif isinstance(cs, ndarray):
            cs = utils.try_to_squeeze_array(cs)
            if cs.dtype.kind == 'b':
                if len(cs) != self.shape[1]:
                    raise ValueError('Length of column selection boolean '
                                     'array must be the same as number of '
                                     'columns in the DataFrame. '
                                     f'{len(rs)} != {self.shape[1]}')
            elif cs.dtype.kind == 'i':
                cs = cs.tolist()
            else:
                raise TypeError('Column selection array data type must be '
                                'either integer or boolean')
            cs = self._get_list_of_cols_from_selection(cs)
        elif isinstance(cs, self.__class__):
            if cs.shape[0] != 1:
                raise ValueError('Boolean selection only works with single-'
                                 'row DataFames')
            cs = cs.values.squeeze()
            if cs.dtype.kind != 'b':
                raise TypeError('All values for column selection must '
                                'be boolean')
            if len(cs) != self.shape[1]:
                raise ValueError('Number of booleans in DataFrame does not '
                                 'equal number of columns in self '
                                 f'{len(cs)} != {self.shape[1]}')
            cs = self._columns[cs]
        else:
            raise TypeError('Selection must either be one of '
                            'int, str, list, array, slice or DataFrame')
        return cs

    def _convert_row_selection(self, rs: RowSelection):
        if isinstance(rs, slice):
            def check_none_int(obj):
                return obj is None or isinstance(obj, int)

            all_ok: bool = (check_none_int(rs.start) and
                            check_none_int(rs.stop) and
                            check_none_int(rs.step))

            if not all_ok:
                raise TypeError('Slice start, stop, and step values must '
                                'be int or None')
        elif isinstance(rs, list):
            for row in rs:
                # self.columns is a list to prevent numpy warning
                if not isinstance(row, int):
                    raise TypeError('Row selection must consist '
                                    'only of integers')
        elif isinstance(rs, ndarray):
            rs = utils.try_to_squeeze_array(rs)
            if rs.dtype.kind == 'b':
                if len(rs) != len(self):
                    raise ValueError('Length of boolean array must be the '
                                     'same as DataFrame. '
                                     f'{len(rs)} != {len(self)}')
            elif rs.dtype.kind != 'i':
                raise TypeError('Row selection array data type must be '
                                'either integer or boolean')
        elif isinstance(rs, self.__class__):
            if rs.shape[1] != 1:
                raise ValueError('Boolean selection only works with single-'
                                 'column DataFames')
            rs = rs.values.squeeze()
            if rs.dtype.kind != 'b':
                raise TypeError('All values for row selection must '
                                'be boolean')
        elif not isinstance(rs, int):
            raise TypeError('Selection must either be one of '
                            'int, list, array, slice, or DataFrame')
        return rs

    def _getitem_scalar(self, rs, cs):
        # most common case, string column, integer row
        try:
            dtype, loc, _ = self._column_dtype[cs].values
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
                dtype, loc, _ = self._column_dtype[cs].values
                return self._data[dtype][rs, loc]

    def _construct_df_from_selection(self, rs, cs) -> 'DataFrame':
        new_data: Dict[str: ndarray] = {}
        dt_positions = {'b': [], 'i': [], 'f': [], 'O': []}
        new_column_dtype = {}
        new_columns = cs

        for i, col in enumerate(cs):
            dtype, loc, _ = self._column_dtype[col].values
            cur_loc = len(dt_positions[dtype])
            new_column_dtype[col] = utils.Column(dtype, cur_loc, i)
            dt_positions[dtype].append(loc)

        for dtype, pos in dt_positions.items():
            if pos:
                if isinstance(rs, (list, ndarray)):
                    ix = np.ix_(rs, pos)
                    arr = np.atleast_2d(self._data[dtype][ix]).copy()
                else:
                    arr = np.atleast_2d(self._data[dtype][rs, pos]).copy()
                if arr.dtype.kind == 'U':
                    arr = arr.astype('O')
                elif arr.dtype.kind == 'S':
                    arr = arr.astype('U').astype('O')
                new_data[dtype] = arr

        return self._construct_from_new(new_data, new_column_dtype,
                                        new_columns)

    def __getitem__(self, value: Tuple[RowSelection,
                                       ColumnSelection]) -> 'DataFrame':
        utils.validate_selection_size(value)
        row_selection, col_selection = value
        if (isinstance(row_selection, int) and
                isinstance(col_selection, (int, str))):
            return self._getitem_scalar(row_selection, col_selection)

        col_selection = self._convert_col_selection(col_selection)
        row_selection = self._convert_row_selection(row_selection)

        return self._construct_df_from_selection(row_selection, col_selection)

    def to_dict(self, orient: str='array'):
        '''
        Conver DataFrame to dictionary of 1-dimensional arrays or lists

        Parameters
        ----------
        orient : str {'array' or 'list'}
        Determines the type of the values of the dictionary.
        '''
        if orient not in ['array', 'list']:
            raise ValueError('orient must be either "array" or "list"')
        data = {}

        for col, col_obj in self._column_dtype.items():
            dtype, loc, order = col_obj.values
            arr = self._data[dtype][:, loc]
            if orient == 'array':
                data[col] = arr.copy()
            else:
                data[col] = arr.tolist()
        return data

    def _is_numeric_or_bool(self):
        return set(self._data.keys()) <= _NUMERIC_KINDS

    def _is_numeric_strict(self):
        return set(self._data.keys()) <= {'i', 'f'}

    def _is_string(self):
        return set(self._data.keys()) == {'O'}

    def _is_only_numeric_or_string(self):
        dtypes = set(self._data.keys())
        return dtypes <= {'i', 'f'} or dtypes == {'O'}

    def _has_numeric_or_bool(self):
        """
        Does this DataFrame have at least one numeric columns?
        """
        dtypes = set(self._data.keys())
        return 'i' in dtypes or 'f' in dtypes or 'b' in dtypes

    def _has_numeric_strict(self):
        """
        Does this DataFrame have at least one numeric columns?
        """
        dtypes = set(self._data.keys())
        return 'i' in dtypes or 'f' in dtypes

    def _has_string(self):
        return 'O' in self._data

    def copy(self):
        new_data = {dt: arr.copy() for dt, arr in self._data.items()}
        new_columns = self._columns.copy()
        new_column_dtype = {col: copy.copy(col_obj)
                            for col, col_obj in self._column_dtype.items()}
        return self._construct_from_new(new_data, new_column_dtype,
                                        new_columns)

    def select_dtypes(self, include=None, exclude=None):
        if include is not None and exclude is not None:
            raise ValueError('Provide only one of either include/exclude')
        if include is None and exclude is None:
            return self.copy()

        if include is None:
            clude = exclude
            arg_name: str = 'exclude'
        if exclude is None:
            clude = include
            arg_name: str = 'include'

        clude = utils.convert_clude(clude, arg_name)

        include_final: List[str] = []
        current_dtypes = set(self._data.keys())
        if arg_name == 'include':
            include_final = [dt for dt in clude if dt in current_dtypes]
        else:
            include_final = [dt for dt in clude if dt not in current_dtypes]

        new_data = {dtype: arr.copy('F')
                    for dtype, arr in self._data.items()
                    if dtype in include_final}

        new_column_dtype = {}
        new_columns = []
        order = 0
        dtype_loc_count = defaultdict(int)
        for col, col_obj in self._column_dtype.items():
            dtype = col_obj.dtype
            if dtype in include_final:
                loc = dtype_loc_count[dtype]
                dtype_loc_count[dtype] += 1
                new_column_dtype[col] = utils.Column(dtype, loc, order)
                new_columns.append(col)
                order += 1

        return self._construct_from_new(new_data, new_column_dtype,
                                        new_columns)

    @classmethod
    def _construct_from_new(cls, data, column_dtype, columns):
        df_new = cls.__new__(cls)
        df_new._column_dtype = column_dtype
        df_new._data = data
        df_new._columns = np.asarray(columns, dtype='O')
        df_new._hasnans = {}
        return df_new

    def _do_eval(self, op_string, other):
        data_dict: Dict[str, ndarray] = {'b': [], 'i': [],
                                         'f': [], 'O': []}
        new_column_dtype = {}
        order = 0
        for dt, arr in self._data.items():
            arr_res = eval(f"{'arr'} .{op_string}({'other'})")
            new_dt = arr_res.dtype.kind
            data_dict[new_dt].append(arr_res)
            cur_len = utils.get_arr_length(data_dict, get(new_dt, []))
            for i, col in enumerate(cols):
                new_column_dtype[col] = utils.Column(new_dt, i + cur_len,
                                                     order)
                order += 1
        return data_dict, new_column_dtype

    def _op(self, other, op_string):
        if isinstance(other, (int, float, bool)):
            if self._is_numeric_or_bool():
                dd, ncd = self._do_eval(op_string, other)
            elif self._is_string():
                if op_string in ['__mul__', '__rmul__']:
                    dd, ncd = self._do_eval(op_string, other)
                else:
                    raise TypeError('DataFrames consisting of all strings '
                                    'work only with the multiplication '
                                    'operator with numerics')
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

        new_data = {dt: np.concatenate(arrs, axis=1)
                    for dt, arrs in dd.items() if arrs}
        return self._construct_from_new(new_data, ncd, self._columns.copy())

    def __add__(self, other):
        return self._op(other, '__add__')

    def __radd__(self, other):
        return self._op(other, '__radd__')

    def __mul__(self, other):
        return self._op(other, '__mul__')

    def __rmul__(self, other):
        return self._op(other, '__rmul__')

    def __sub__(self, other):
        return self._op(other, '__sub__')

    def __rsub__(self, other):
        return self._op(other, '__rsub__')

    def __truediv__(self, other):
        return self._op(other, '__truediv__')

    def __rtruediv__(self, other):
        return self._op(other, '__rtruediv__')

    def __floordiv__(self, other):
        return self._op(other, '__floordiv__')

    def __rfloordiv__(self, other):
        return self._op(other, '__rfloordiv__')

    def __pow__(self, other):
        return self._op(other, '__pow__')

    def __rpow__(self, other):
        return self._op(other, '__rpow__')

    def __mod__(self, other):
        return self._op(other, '__mod__')

    def __rmod__(self, other):
        return self._op(other, '__rmod__')

    def __gt__(self, other):
        return self._op(other, '__gt__')

    def __ge__(self, other):
        return self._op(other, '__ge__')

    def __lt__(self, other):
        return self._op(other, '__lt__')

    def __le__(self, other):
        return self._op(other, '__le__')

    def __eq__(self, other):
        return self._op(other, '__eq__')

    def __ne__(self, other):
        return self._op(other, '__ne__')

    def __neg__(self):
        if self._is_numeric_or_bool():
            new_data = {}
            for dt, arr in self._data.items():
                new_data[dt] = -arr.copy()
        else:
            raise ValueError('Only works for all numeric columns')
        new_column_dtype = {col: utils.Column(*col_obj.values)
                            for col, col_obj in self._column_dtype.items()}
        return self._construct_from_new(new_data, new_column_dtype,
                                        self._columns.copy())

    def __bool__(self):
        raise ValueError(': The truth value of an array with more than '
                         'one element is ambiguous. '
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

    def __iadd__(self, value):
        raise NotImplementedError(f'Use df = df + {value}')

    def __isub__(self, value):
        raise NotImplementedError(f'Use df = df - {value}')

    def __imul__(self, value):
        raise NotImplementedError(f'Use df = df * {value}')

    def __idiv__(self, value):
        raise NotImplementedError(f'Use df = df / {value}')

    def __ifloordiv__(self, value):
        raise NotImplementedError(f'Use df = df // {value}')

    def __imod__(self, value):
        raise NotImplementedError(f'Use df = df % {value}')

    def __ipow__(self, value):
        raise NotImplementedError(f'Use df = df ** {value}')

    @property
    def dtypes(self):
        arr = [utils.convert_kind_to_dtype(self._column_dtype[col].dtype)
               for col in self._columns]
        arr = np.array(arr, dtype='O')
        columns = ['Column Name', 'Data Type']
        cn = self._columns.astype('O')
        data = np.column_stack((cn, arr))
        new_data: Dict[str, ndarray] = {'O': data}
        new_column_dtype = {'Column Name': utils.Column('O', 0, 0),
                            'Data Type': utils.Column('O', 1, 1)}
        return self._construct_from_new(new_data, new_column_dtype, columns)

    def __and__(self, other):
        return self._op_logical(other, '__and__')

    def __or__(self, other):
        return self._op_logical(other, '__or__')

    def _validate_matching_shape(self, other):
        if isinstance(other, self.__class__):
            oshape = other.shape
        elif isinstance(other, ndarray):
            if other.ndim == 1:
                oshape = (len(other), 1)
            else:
                oshape = other.shape

        if self.shape != other.shape:
            raise ValueError('Shape of left DataFrame does not match '
                             'shape of right'
                             f'{self.shape} != {other.shape}')

    def _op_logical(self, other, op_logical):
        if not isinstance(other, (self.__class__, ndarray)):
            d = {'__and__': '&', '__or__': '|'}
            raise TypeError(f'Must use {d[op_logical]} operator with either '
                            ' DataFrames or arrays.')

        self._validate_matching_shape(other)
        is_other_df = isinstance(other, self.__class__)
        new_data = {}

        arr_left = self.values

        if is_other_df:
            arr_right = other.values
        else:
            arr_right = other

        arr_res = getattr(arr_left, op_logical)(arr_right)
        new_data[arr_res.dtype.kind] = arr_res
        new_column_dtype = {col: utils.Column('b', i, i)
                            for i, (col, _) in
                            enumerate(self._column_dtype.items())}

        return self._construct_from_new(new_data, new_column_dtype,
                                        self._columns.copy())

    def __invert__(self):
        if set(self._data) == {'b'}:
            new_data = {dt: ~arr for dt, arr in self._data.items()}
            new_column_dtype = self._copy_cd()
            return self._construct_from_new(new_data,
                                            new_column_dtype,
                                            self._columns.copy())
        else:
            raise TypeError('Invert operator only works on DataFrames '
                            'with all boolean columns')

    def _copy_cd(self):
        return {col: utils.Column(*col_obj.values)
                for col, col_obj in self._column_dtype.items()}

    def _astype_internal(self, column, numpy_dtype):
        """
        Changes one column dtype in-place
        """
        new_kind = utils.convert_numpy_to_kind(numpy_dtype)
        dtype, loc, order = self._column_dtype[column].values
        if dtype == new_kind:
            return None
        col_data = self._data[dtype][:, loc]
        if numpy_dtype == 'O':
            col_data = col_data.astype('U').astype('O')
        else:
            col_data = col_data.astype(numpy_dtype)
        self._remove_column(column)
        self._write_new_column_data(column, new_kind, col_data, order)

    def _remove_column(self, column):
        """
        Removes column from _colum_dtype, and _data
        Keeps column name in _columns
        """
        dtype, loc, order = self._column_dtype.pop(column).values
        self._data[dtype] = np.delete(self._data[dtype], loc, axis=1)
        for col, col_obj in self._column_dtype.items():
            if col_obj.dtype == dtype and col_obj.loc > loc:
                col_obj.loc -= 1

    def _write_new_column_data(self, column, new_kind, data, order):
        """
        Adds data to _data, the data type info but does not
        append name to columns
        """
        loc = self._data[new_kind].shape[1]
        self._column_dtype[column] = utils.Column(new_kind, loc, order)
        if new_kind in self._data:
            self._data[new_kind] = np.column_stack((self._data[new_kind],
                                                    data))
        else:
            self._data[new_kind] = data

    def _add_new_column(self, column, kind, data):
        order = len(self._columns)
        self._write_new_column_data(column, kind, data, order)
        self._columns = np.append(self._columns, column)

    def _full_columm_add(self, column, kind, data):
        """
        Either adds a brand new column or
        overwrites an old column
        """
        if column not in self._column_dtype:
            self._add_new_column(column, kind, data)
        # column is in df
        else:
            # data type has changed
            dtype, loc, order = self._column_dtype[column].values
            if dtype != kind:
                self._remove_column(column)
                self._write_new_column_data(column, kind, data, order)
            # data type same as original
            else:
                self._data[kind][:, loc] = data

    def __setitem__(self, key, value):
        utils.validate_selection_size(key)

        # row selection and column selection
        rs, cs = key
        if isinstance(rs, int) and isinstance(cs, (int, str)):
            return self._setitem_scalar(rs, cs, value)

        # select an entire column or new column
        if utils.is_entire_column_selection(rs, cs):
            return self._setitem_entire_column(rs, cs, value)

        cs = self._convert_col_selection(cs)
        rs = self._convert_row_selection(rs)

        self._setitem_all_other(rs, cs, value)

    def _setitem_scalar(self, rs, cs, value):
        """
        Assigns a scalar to exactly a single cell
        """
        if isinstance(cs, str):
            self._validate_column_name(cs)
        else:
            cs = self._get_col_name_from_int(cs)
        dtype, loc, order = self._column_dtype[cs].values

        if isinstance(value, bool):
            utils.check_set_value_type(dtype, 'b', 'bool')
        elif isinstance(value, (int, np.integer)):
            utils.check_set_value_type(dtype, 'if', 'int')
        elif isinstance(value, (float, np.floating)):
            utils.check_set_value_type(dtype, 'if', 'float')
            if dtype == 'i':
                self._astype_internal(cs, np.float64)
        elif isinstance(value, str):
            utils.check_set_value_type(dtype, 'O', 'str')
        elif isinstance(value, bytes):
            value = value.decode()
            utils.check_set_value_type(dtype, 'O', 'bytes')
        else:
            raise TypeError(f'Type {type(value).__name__} not able '
                            'to be assigned')
        self._data[dtype][rs, loc] = value

    def _setitem_entire_column(self, rs, cs, value):
        """
        Called when setting an entire column (old or new)
        df[:, 'col'] = value
        """
        if utils.is_scalar(value):
            data = np.repeat(value, len(self))
            data = utils.convert_bytes_or_unicode(data)
            kind = data.dtype.kind
            self._full_columm_add(cs, kind, data)
        elif isinstance(value, (ndarray, list)):
            if isinstance(value, list):
                value = utils.maybe_convert_1d_array(np.array(value))
            value = utils.try_to_squeeze_array(value)
            utils.validate_array_size(value, len(self))
            if value.dtype.kind == 'O':
                va.validate_strings_in_object_array(value)
            self._full_columm_add(cs, value.dtype.kind, value)
        else:
            raise TypeError('Must use a scalar, a list, an array, or a '
                            'DataFrame when setting new values')

    def _setitem_all_other(self, rs, cs, value):
        """
        Sets new data when not assigning a scalar
        and not assigning a single column
        """
        value_kind = utils.get_kind_from_scalar(value)
        if value_kind:
            self._validate_setitem_col_types(cs, value_kind)
            for col in cs:
                dtype, loc, order = self._column_dtype[col].values
                if dtype == 'i' and value_kind == 'f':
                    self._astype_internal(col, np.float64)
                    dtype, loc, order = self._column_dtype[col].values
                self._data[dtype][rs, loc] = value
        # not scalar
        else:
            if isinstance(value, list):
                arrs = utils.convert_list_to_arrays(value)
            elif isinstance(value, ndarray):
                arrs = utils.convert_array_to_arrays(value)
            elif isinstance(value, self.__class__):
                arrs = []
                for col, col_obj in self._column_dtype.items():
                    dtype, loc, order = col_obj.values
                    arrs.append(self._data[dtype][:, loc])
            else:
                raise TypeError('Must use a scalar, a list, an array, or a '
                                'DataFrame when setting new values')
            self._validate_set_array_shape(rs, cs, arrs)
            kinds, other_kinds = self._get_kinds(cs, arrs)
            utils.check_compatible_kinds(kinds, other_kinds)
            self._setitem_other_cols(rs, cs, arrs, kinds, other_kinds)

    def _validate_setitem_col_types(self, columns, kind):
        """
        Used to verify column dtypes when setting a scalar
        to many columns
        """
        for col in columns:
            cur_kind = self._column_dtype[col].dtype
            if cur_kind == kind or (cur_kind in 'if' and kind in 'if'):
                continue
            else:
                dt = utils.convert_kind_to_dtype(kind)
                ct = utils.convert_kind_to_dtype(cur_kind)
                raise TypeError(f'Trying to set a {dt} on column'
                                f' {col} which has type {ct}')

    def _validate_set_array_shape(self, rows, cols, other):
        num_rows_to_set = len(np.arange(len(self))[rows])
        num_cols_to_set = len(cols)

        if isinstance(other, list):
            num_rows_set = len(other[0])
            num_cols_set = len(other)
        # Otherwise we have a DataFrame
        else:
            num_rows_set = other.shape[0]
            num_cols_set = other.shape[1]

        if num_rows_to_set != num_rows_set:
            raise ValueError('Mismatch of number of rows '
                             f'{num_rows_to_set} != {num_rows_set}')
        if num_cols_to_set != num_cols_set:
            raise ValueError('Mismatch of number of columns'
                             f'{num_cols_to_set} != {num_cols_set}')

    def _get_kinds(self, cols, other):
        kinds = [self._column_dtype[col].dtype for col in cols]
        if isinstance(other, list):
            other_kinds = [arr.dtype.kind for arr in other]
        else:
            other_kinds = [other._column_dtype[col].dtype
                           for col in other._columns]
        return kinds, other_kinds

    def _setitem_other_cols(self, rows, cols, arrs, kinds1, kinds2):
        for col, arr, k1, k2 in zip(cols, arrs, kinds1, kinds2):
            if k1 == 'i' and k2 == 'f':
                dtype = utils.convert_kind_to_numpy(k2)
                self._astype_internal(col, dtype)
            dtype, loc, order = self._column_dtype[col]
            self._data[dtype][rows, loc] = arr

    def _validate_column_name(self, column):
        if column not in self._column_dtype:
            raise ValueError(f'Column {column} does not exist')

    def _new_cd_from_kind_shape(self, kind_shape, new_kind):
        new_column_dtype = {}
        for col, col_obj in self._column_dtype.items():
            dtype, loc, order = col_obj.values
            add_loc = kind_shape[dtype]
            new_column_dtype[col] = utils.Column(new_kind, loc + add_loc,
                                                 order)
        return new_column_dtype

    def astype(self, dtype):
        if isinstance(dtype, str):
            new_dtype = utils.check_valid_dtype_convet(dtype)
            data_dict = defaultdict(list)
            kind_shape = OrderedDict()
            total_shape = 0
            for old_kind, arr in self._data.items():
                kind_shape[old_kind] = total_shape
                total_shape += arr.shape[1]
                arr = arr.astype(new_dtype)
                if arr.dtype.kind == 'U':
                    arr = arr.astype('O')
                new_kind = arr.dtype.kind
                data_dict[new_kind].append(arr)

            new_column_dtype = self._new_cd_from_kind_shape(kind_shape, new_kind)
            new_data = utils.concat_stat_arrays(data_dict)
            return self._construct_from_new(new_data, new_column_dtype,
                                            self._columns.copy())

        elif isinstance(dtype, dict):
            df_new = self.copy()
            for column, new_dtype in dtype.items():
                df_new._validate_column_name(column)
                new_dtype_numpy = utils.check_valid_dtype_convet(new_dtype)
                df_new._astype_internal(column, new_dtype_numpy)
            return df_new
        else:
            raise TypeError('Argument dtype must be either a string '
                            'or a dictionary')

    def head(self, n=5):
        return self[:n, :]

    def tail(self, n=5):
        return self[-n:, :]

    @property
    def hasnans(self):
        if self._hasnans == {}:
            for kind, arr in self._data.items():
                kinds += kind
                if kind == 'f':
                    self._hasnans['f'] = np.isnan(arr).any(0)
                elif kind == 'O':
                    self._hasnans['O'] = va.isnan_object(arr)
                else:
                    self._hasnans[kind] = np.zeros(arr.shape[1], dtype='bool')

        bool_array = np.empty(len(self), dtype='bool')

        for col, col_obj in self._column_dtype.items():
            kind, loc, order = col_obj.values
            bool_array[order] = self._hasnans[kind][loc]

        columns = np.array(['Column Name', 'Has NaN'])
        new_data = {'O': self._columns[:, np.newaxis],
                    'b': bool_array[col_order, np.newaxis]}

        new_column_dtype = {'Column Name': utils.Column('O', 0, 0),
                            'Has NaN': utils.Column('b', 0, 1)}
        return self._construct_from_new(new_data, new_column_dtype, columns)

    def _get_return_dtype(self, name):
        if name in ['count', 'argmax', 'argmin']:
            return np.int64
        elif name in ['std', 'var']:
            return np.float64
        elif name in ['cummax', 'cummin', 'cumsum', 'sum', 'max', 'min']:
            if self._data.keys() == {'i'}:
                return np.int64
            return np.float64
        # any and all need to have special row wise any_obj and all_obj
        elif name in ['any', 'all']:
            return bool

    def _get_specific_stat_data(self, name, axis):
        if axis == 0:
            if name in ['std', 'var', 'mean', 'median']:
                return self._get_numeric()
        elif axis == 1:
            if name in ['std', 'var', 'mean', 'median']:
                return self._get_numeric()
            elif name in ['sum', 'max', 'min', 'cumsum', 'cummax', 'cummin',
                          'argmin', 'argmax']:
                if self._is_numeric_or_bool():
                    return self
                elif self._is_string():
                    return self
                return self._get_numeric()
        return self

    def _get_stat_func_result(self, func, arr, axis, kwargs):
        result = func(arr, axis=axis, **kwargs)

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

    def _get_kind_cols(self):
        kind_cols = defaultdict(list)
        for col, col_obj in df._column_dtype.items():
            kind_cols[col_obj.dtype].append(col)

    def _stat_funcs(self, name, axis, **kwargs):
        axis = utils.convert_axis_string(axis)
        data_dict = defaultdict(list)
        df = self._get_specific_stat_data(name, axis)
        new_column_dtype = {}

        change_kind = {}
        if axis == 0:
            for kind, arr in df._data.items():
                func = stat.funcs[kind][name]
                hasnans = df._hasnans_dtype(kind)
                kwargs.update({'hasnans': hasnans})
                arr = df._get_stat_func_result(func, arr, 0, kwargs)
                del kwargs['hasnans']

                new_kind = arr.dtype.kind
                cur_loc = utils.get_arr_length(data_dict.get(new_kind, []))
                change_kind[kind] = (new_kind, cur_loc)
                data_dict[new_kind].append(arr)

            for col, col_obj in df._column_dtype.items():
                kind, loc, order = col_obj.values
                new_kind, add_loc = change_kind[kind]
                new_column_dtype[col] = utils.Column(new_kind,
                                                     loc + add_loc, order)

            new_data = utils.concat_stat_arrays(data_dict)
            return df._construct_from_new(new_data, new_column_dtype,
                                          df._columns.copy())
        else:
            arrs = []
            if utils.is_column_stack_func(name):
                arr = df._values_raw()
                kind = arr.dtype.kind
                hasnans = df._hasnans_dtype(kind)
                kwargs.update({'hasnans': hasnans})
                func = stat.funcs[kind][name]
                result = df._get_stat_func_result(func, arr, 1, kwargs)
                del kwargs['hasnans']
            else:
                for kind, arr in df._data.items():
                    func = stat.funcs[kind][name]
                    hasnans = df._hasnans_dtype(kind)
                    kwargs.update({'hasnans': hasnans})
                    arr = df._get_stat_func_result(func, arr, 1, kwargs)
                    del kwargs['hasnans']
                    arrs.append(arr)

                if len(arrs) == 1:
                    result = arrs[0]
                else:
                    func_across = stat.funcs_columns[name]
                    result = func_across(arrs)

            if utils.is_agg_func(name):
                new_columns = np.array([name], dtype='O')
            else:
                new_columns = df._columns.copy()

            new_kind = result.dtype.kind
            new_data = {new_kind: result}

            for i, col in enumerate(new_columns):
                new_column_dtype[col] = utils.Column(new_kind, i, i)
            return df._construct_from_new(new_data, new_column_dtype,
                                          new_columns)

    def sum(self, axis='rows'):
        return self._stat_funcs('sum', axis)

    def max(self, axis='rows'):
        return self._stat_funcs('max', axis)

    def min(self, axis='rows'):
        return self._stat_funcs('min', axis)

    def mean(self, axis='rows'):
        return self._stat_funcs('mean', axis)

    def median(self, axis='rows'):
        return self._stat_funcs('median', axis)

    def std(self, axis='rows', ddof=1):
        return self._stat_funcs('std', axis, ddof=ddof)

    def var(self, axis='rows', ddof=1):
        return self._stat_funcs('var', axis, ddof=ddof)

    def abs(self):
        df = self._get_numeric()
        new_data = {dt: np.abs(arr) for dt, arr in df._data.items()}
        new_column_dtype = self._copy_cd()
        return self._construct_from_new(new_data, new_column_dtype,
                                        df._columns.copy())

    __abs__ = abs

    def _get_numeric(self):
        if not self._has_numeric_or_bool():
            raise TypeError('All columns must be either integer, '
                            'float, or boolean')
        return self.select_dtypes(['number', 'bool'])

    def _check_if_hasnans_exist(self):
        if hasattr(self, 'hasnans'):
            self.hasnans.any()
        else:
            pass

    def any(self, axis='rows'):
        return self._stat_funcs('any', axis)

    def all(self, axis='rows'):
        return self._stat_funcs('all', axis)

    def argmax(self, axis='rows'):
        return self._stat_funcs('argmax', axis)

    def argmin(self, axis='rows'):
        return self._stat_funcs('argmin', axis)

    def count(self, axis='rows'):
        return self._stat_funcs('count', axis)

    def _get_clip_df(self, value, name):
        if value is None:
            raise ValueError('You must provide a value for either lower '
                             'or upper')
        if utils.is_number(value):
            if self._has_numeric_or_bool():
                return self.select_dtypes(['number', 'bool']), 'number'
            else:
                raise TypeError(f'You provided a numeric value for {name} '
                                'but do not have any numeric columns')
        elif isinstance(value, str):
            if self._has_string():
                return self.select_dtypes('str'), 'str'
            else:
                raise TypeError(f'You provided a string value for {name} '
                                'but do not have any string columns')
        else:
            raise NotImplementedError('Data type incompatible')

    def clip(self, lower=None, upper=None):
        if lower is None:
            df, overall_dtype = self._get_clip_df(upper, 'upper')
        elif upper is None:
            df, overall_dtype = self._get_clip_df(lower, 'lower')
        else:
            overall_dtype = utils.is_compatible_values(lower, upper)
            if lower > upper:
                raise ValueError('The upper value must be less than lower')
            if overall_dtype == 'number':
                df = self.select_dtypes(['number', 'bool'])
            else:
                df = self.select_dtypes('str')

        if overall_dtype == 'str':
            new_data = {}
            if lower is None:
                lower = chr(0)
            if upper is None:
                upper = chr(1000000)
            new_data['O'] = _math.clip_str(df._data['O'], lower, upper)
        else:
            if utils.is_integer(lower) or utils.is_integer(upper):
                if 'b' in self._data:
                    as_type_dict = {col: 'int'
                                    for col in df._dtype_column['b']}
                    df = df.astype(as_type_dict)
            if utils.is_float(lower) or utils.is_float(upper):
                if 'i' in df._data:
                    as_type_dict = {col: 'float'
                                    for col in df._dtype_column['i']}
                    df = df.astype(as_type_dict)
                if 'b' in df._data:
                    as_type_dict = {col: 'float'
                                    for col in df._dtype_column['b']}
                    df = df.astype(as_type_dict)
            new_data = {}
            for dtype, arr in df._data.items():
                new_data[dtype] = arr.clip(lower, upper)

        new_column_dtype = df._copy_cd()
        return df._construct_from_new(new_data, new_column_dtype,
                                      df._columns.copy())

    def cummax(self, axis='rows'):
        return self._stat_funcs('cummax', axis)

    def cummin(self, axis='rows'):
        return self._stat_funcs('cummin', axis)

    def cumsum(self, axis='rows'):
        return self._stat_funcs('cumsum', axis)

    def _hasnans_dtype(self, kind):
        hasnans = np.ones(self.shape[1], dtype='bool')
        return self._hasnans.get(kind, hasnans)

    def isna(self):
        data_dict = defaultdict(list)
        new_column_dtype = {}
        n = len(self)
        i = 0

        kind_shape = OrderedDict()
        total_shape = 0
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

        new_column_dtype = self._new_cd_from_kind_shape(kind_shape, 'b')
        new_data = utils.concat_stat_arrays(data_dict)
        return self._construct_from_new(new_data, new_column_dtype,
                                        self._columns.copy())

    def describe(self):
        df = self.select_dtypes('number')
        num_cols = df.shape[1]
        new_data = {'Column Name': df._columns.astype('O'),
                    'count': np.empty(num_cols, dtype=np.int64),
                    'mean': np.empty(num_cols, dtype=np.float64),
                    'std': np.empty(num_cols, dtype=np.float64),
                    'min': np.empty(num_cols, dtype=np.float64),
                    '25%': np.empty(num_cols, dtype=np.float64),
                    '50%': np.empty(num_cols, dtype=np.float64),
                    '75%': np.empty(num_cols, dtype=np.float64),
                    'max': np.empty(num_cols, dtype=np.float64)}
        new_columns = np.array(['Column Name', 'count', 'mean', 'std', 'min',
                                '25%', '50%', '75%', 'max'])
        for i, col in enumerate(df._columns):
            arr = df._data[col]
            kind = arr.dtype.kind
            for name in new_columns[1:]:
                func = stat.funcs[kind][1].get(name)
                if func is not None:
                    new_data[name][i] = func(arr)
                else:
                    new_data[name][i] = -999

        return self._construct_from_new(new_data, new_columns)

    def dropna(self, axis='rows', how='any', thresh=None, subset=None):
        """
        thresh overtakes any how and can either be an integer
        or a float between 0 and 1
        """
        axis = utils.swap_axis_name(axis)
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
                    raise ValueError('thresh must either be an integer or '
                                     ' a float between 0 and 1')
                criteria = (~df.isna()).mean(axis) >= thresh
            else:
                raise TypeError('thresh must be an integer or a float '
                                'between 0 and 1')
        elif how == 'any':
            criteria = ~df.isna().any(axis)
        elif how == 'all':
            criteria = ~df.isna().all(axis)
        else:
            raise ValueError('how must be either "any" or "all"')

        if axis == 'rows':
            return self[:, criteria]
        return self[criteria, :]

    def cov(self, method='pearson', min_periods=1):
        df = self.select_dtypes(['number', 'bool'])
        values = df._values_number()
        n = df.shape[1]
        cov_final = np.empty((n, n), dtype=np.float64)

        for i in range(n):
            for j in range(i, n):
                curr_cov = _math.cov(values[:, i], values[:, j])
                cov_final[i, j] = curr_cov
                cov_final[j, i] = curr_cov

        new_data = {'f': np.asfortranarray(cov_final)}
        new_column_dtype = {'Column Name': utils.Column('O', 0, 0)}
        new_columns = np.empty(df.shape[1] + 1, dtype='O')
        new_columns[0] = 'Column Name'
        for i, col in enumerate(df._columns):
            new_column_dtype[col] = utils.Column('f', i, i + 1)
            new_columns[i + 1] = col
        new_data['O'] = np.asfortranarray(df._columns[:, np.newaxis])
        return df._construct_from_new(new_data, new_column_dtype, new_columns)

    def corr(self, method='pearson', min_periods=1):
        df = self.select_dtypes(['number', 'bool'])
        values = df._values_number()
        n = df.shape[1]
        corr_final = np.empty((n, n), dtype=np.float64)
        # np.fill_diagonal(corr_final, 1)

        for i in range(n):
            for j in range(i, n):
                curr_corr = _math.corr(values[:, i], values[:, j])
                corr_final[i, j] = curr_corr
                corr_final[j, i] = curr_corr

        new_data = {'f': np.asfortranarray(corr_final)}
        new_column_dtype = {'Column Name': utils.Column('O', 0, 0)}
        new_columns = np.empty(df.shape[1] + 1, dtype='O')
        new_columns[0] = 'Column Name'
        for i, col in enumerate(df._columns):
            new_column_dtype[col] = utils.Column('f', i, i + 1)
            new_columns[i + 1] = col
        new_data['O'] = np.asfortranarray(df._columns[:, np.newaxis])
        return df._construct_from_new(new_data, new_column_dtype, new_columns)

    def unique(self, col):
        self._validate_column_name(col)
        arr = self._get_col_array(col)
        kind = arr.dtype.kind
        if kind == 'i':
            amin, amax = _math.min_max_int(arr)
            if amax - amin < 10_000_000:
                return _math.unique_bounded(arr, amin)
            else:
                return _math.unique_int(arr)
        elif kind == 'f':
            return _math.unique_float(arr)
        elif kind == 'b':
            return _math.unique_bool(arr)
        return _math.unique_object(arr)

    def value_counts(self, col):
        self._validate_column_name(col)
        arr = self._get_col_array(col)

        (group_labels,
         group_names,
         group_position) = gb.get_group_assignment(arr)

        size = gb.size(group_labels, d)

        new_data: Block = {'O': group_names,
                           'b': None, 'f': None, 'i': size[:, np.newaxis]}
        new_columns = ['Column Values', 'Counts']
        # add _column_dtype
        return self._construct_from_new(new_data, new_columns)

    def groupby(self, columns):
        if isinstance(columns, list):
            for col in columns:
                self._validate_column_name(col)
        elif isinstance(columns, str):
            self._validate_column_name(columns)
        else:
            raise ValueError('Must pass in grouping column(s) as a string '
                             'or list of strings')

        return Grouper(self, columns)


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
        # add _column_dtype
        return self.df._construct_from_new(new_data, new_columns)


class Grouper(object):

    def __init__(self, df, columns):
        self._df = df
        self._create_groups(columns)
        self._group_columns = columns

    def _create_groups(self, columns):
        arr = self._df._get_col_array(columns)
        (self._group_labels,
         self._group_names,
         self._group_position) = gb.get_group_assignment(arr)

    def size(self):
        size = gb.size(self._group_labels, len(self._group_names))
        df_new = self._df[self._group_position, self._group_columns]
        df_new[:, 'size'] = size
        return df_new
