from dexplo._frame import DataFrame
import dexplo._utils as utils
from dexplo._libs import (string_funcs as _sf,
                          math as _math)
import re
import numpy as np
from numpy import nan, ndarray
from typing import (Union, Dict, List, Optional, Tuple, Callable, overload,
                    NoReturn, Set, Iterable, Any, TypeVar, Type, Generator)
from typing import Pattern


class StringClass(object):

    def __init__(self, df: DataFrame) -> None:
        self._df: DataFrame = df

    def _validate_columns(self, column):
        if isinstance(column, str):
            try:
                dtype, loc, _ = self._df._column_info[column].values
            except KeyError:
                raise KeyError(f'Column "{column}" does not exist in the DataFrame')
            if dtype != 'O':
                raise ValueError(f'Column name "{column}" is not a str column')
            return [column], [loc]
        elif isinstance(column, list):
            locs = []
            for col in column:
                try:
                    dtype, loc, _ = self._df._column_info[col].values
                except KeyError:
                    raise KeyError(f'Column {col} does not exist in the DataFrame')

                if dtype != 'O':
                    raise ValueError(f'Column name "{col}" is not a str column')
                locs.append(self._df._column_info[col].loc)
            if len(column) != len(set(column)):
                raise ValueError('You cannot complete this operation with duplicate columns')
            return column, locs
        elif column is None:
            locs = []
            columns = []
            for col in self._df._columns:
                dtype, loc, _ = self._df._column_info[col].values
                if dtype == 'O':
                    columns.append(col)
                    locs.append(loc)
            return columns, loc
        else:
            raise TypeError('`column` must be a column name as a string, a list of string, or None')

    def _validate_columns_others(self, column):
        if isinstance(column, str):
            column = [column]

        str_cols = []
        str_locs = []
        for col in self._df._columns:
            dtype, loc, _ = self._df._column_info[col].values
            if dtype == 'O':
                str_cols.append(col)
                str_locs.append(loc)

        if column is None:
            return str_cols, str_locs, [], []

        if isinstance(column, list):
            locs = []
            for col in column:
                try:
                    dtype, loc, _ = self._df._column_info[col].values
                except KeyError:
                    raise KeyError(f'Column {col} does not exist in the DataFrame')

                if dtype != 'O':
                    raise ValueError(f'Column name "{col}" is not a str column')

                locs.append(self._df._column_info[col].loc)
            col_set = set(column)
            if len(column) != len(col_set):
                raise ValueError('You cannot complete this operation with duplicate columns')

            other_cols = []
            other_locs = []
            for col, loc in zip(str_cols, str_locs):
                if col not in col_set:
                    other_cols.append(col)
                    other_locs.append(loc)

            return column, locs, other_cols, other_locs
        else:
            raise TypeError('`column` must be a column name as a string, a list of string, or None')

    def _create_df(self, arr, dtype, columns):
        new_data = {dtype: arr}
        new_column_info = {col: utils.Column(dtype, i, i) for i, col in enumerate(columns)}
        return self._df._construct_from_new(new_data, new_column_info, columns)

    def _create_df_all(self, arr, dtype):
        new_data = {}
        if dtype == 'O':
            for dtype, arr in self._df._data.items():
                if dtype == 'O':
                    new_data['O'] = arr
                new_data[dtype] = arr.copy('F')
        else:
            new_data = {}
            add_loc = 0
            if dtype in self._data:
                add_loc = self._data[dtype].shape[1]
            for old_dtype, old_data in self._df._data.items():
                if dtype != 'O':
                    new_data[old_dtype] = old_data.copy('F')

            if dtype in new_data:
                new_data[dtype] = np.asfortranarray(np.column_stack((new_data[dtype], arr)))
            else:
                new_data[dtype] = arr

            new_column_info = {}
            for col, col_obj in self._df._column_info.items():
                old_dtype, loc, order = col_obj.values
                if old_dtype == 'O':
                    new_column_info[col] = utils.Column(dtype, loc + add_loc, order)
                else:
                    new_column_info[col] = utils.Column(old_dtype, loc, order)

        new_column_info = self._df._copy_column_info()
        return self._df._construct_from_new(new_data, new_column_info, self._df._columns.copy())

    def _create_df_multiple_dtypes(self, arr_new, columns, column_locs, columns_other, locs_other):
        new_data = {}
        dtype_new = arr_new.dtype.kind
        try:
            add_loc = self._df._data[dtype_new].shape[1]
        except KeyError:
            add_loc = 0
        for dtype, arr in self._df._data.items():
            if dtype == 'O':
                new_data['O'] = arr[:, locs_other]
            elif dtype == dtype_new:
                new_data[dtype_new] = np.asfortranarray(np.column_stack((arr, arr_new)))
            else:
                new_data[dtype] = arr.copy('F')

        if dtype_new not in new_data:
            new_data[dtype_new] = arr_new

        new_column_info = {}
        for col, col_obj in self._df._column_info.items():
            old_dtype, loc, order = col_obj.values
            if old_dtype != 'O':
                new_column_info[col] = utils.Column(old_dtype, loc, order)

        # str columns that have changed type
        for i, (col, loc) in enumerate(zip(columns, column_locs)):
            order = self._df._column_info[col].order
            new_column_info[col] = utils.Column(dtype_new, add_loc + i, order)

        # those that stayed str
        for i, col in enumerate(columns_other):
            order = self._df._column_info[col].order
            new_column_info[col] = utils.Column('O', i, order)

        return self._df._construct_from_new(new_data, new_column_info, self._df._columns.copy())

    def _get_other_str_cols(self, columns):
        col_set = set(columns)
        cols_other = []
        locs_other = []
        for col in self._df._columns:
            if col not in col_set and self._df._column_info[col].dtype == 'O':
                cols_other.append(col)
                locs_other.append(self._df._column_info[col].loc)
        return cols_other, locs_other

    def capitalize(self, column=None, keep=False):
        columns, locs = self._validate_columns(column)
        data = self._df._data['O']
        if len(locs) == 1:
            arr = _sf.capitalize(data[:, locs[0]])[:, np.newaxis]
        else:
            arr = _sf.capitalize_2d(data[:, locs])
        if keep:
            data = data.copy()
            for i, loc in enumerate(locs):
                data[:, loc] = arr[:, i]
            return self._create_df_all(data, 'O')
        return self._create_df(arr, 'O', columns)

    def cat(self, column=None):
        columns, locs = self._validate_columns(column)
        if len(locs) == 1:
            arr = self._df._data['O'][:, locs[0]]
            nans = _math.isna_str_1d(arr)
            arr = arr[~nans].tolist()
            return ''.join(arr)
        else:
            data = {}
            for col, loc in zip(columns, locs):
                arr = self._df._data['O'][:, loc]
                nans = _math.isna_str_1d(arr)
                arr = arr[~nans].tolist()
                data[col] = [''.join(arr)]
            return DataFrame(data)

    def center(self, column=None, width=None, fill_character=' ', keep=False):
        if not isinstance(fill_character, str):
            raise TypeError('`fill_character` must be a string')
        elif len(fill_character) != 1:
            raise ValueError('`fill_character` must be exactly one character long')
        if not isinstance(width, (int, np.integer)):
            raise TypeError('`width` must be an integer')

        columns, locs = self._validate_columns(column)
        data = self._df._data['O']

        if len(locs) == 1:
            arr = _sf.center(data[:, locs[0]], width, fill_character)[:, np.newaxis]
        else:
            arr = _sf.center_2d(data[:, locs], width, fill_character)

        if keep:
            data = data.copy()
            for i, loc in enumerate(locs):
                data[:, loc] = arr[:, i]
            return self._create_df_all(data, 'O')
        return self._create_df(arr, 'O', columns)

    def contains(self, column=None, pat=None, case=True, flags=0, na=nan, regex=True, keep=False):
        if not isinstance(case, (bool, np.bool_)):
            raise TypeError('`case` must be a boolean')
        if not isinstance(flags, (int, np.integer, re.RegexFlag)):
            raise TypeError('flags must be a `RegexFlag` or integer')
        if not isinstance(pat, (str, Pattern)):
            raise TypeError('`pat` must either be either a string or compiled regex pattern')
        if not isinstance(regex, (bool, np.bool_)):
            raise TypeError('`regex` must be a boolean')
        if not isinstance(keep, (bool, np.bool_)):
            raise TypeError('`keep` must be a boolean')

        if keep:
            columns, locs, other_columns, other_locs = self._validate_columns_others(column)
        else:
            columns, locs = self._validate_columns(column)

        data = self._df._data['O']
        if len(locs) == 1:
            arr = _sf.contains(data[:, locs[0]], pat, case, flags, na, regex)[:, np.newaxis]
        else:
            arr = _sf.contains_2d(data[:, locs], pat, case, flags, na, regex)

        if keep:
            return self._create_df_multiple_dtypes(arr, columns, locs, other_columns, other_locs)
        else:
            return self._create_df(arr, 'O', columns)

    def count(self, column=None, pat=None, case=True, flags=0, na=nan, regex=True, keep=False):
        """

        Parameters
        ----------
        column
        pat
        case - gets ignored whenever
        flags
        na
        regex
        keep

        Returns
        -------

        """
        if not isinstance(case, (bool, np.bool_)):
            raise TypeError('`case` must be a boolean')
        if not isinstance(flags, (int, np.integer, re.RegexFlag)):
            raise TypeError('flags must be a `RegexFlag` or integer')
        if not isinstance(pat, (str, Pattern)):
            raise TypeError('`pat` must either be either a string or compiled regex pattern')
        if not isinstance(regex, (bool, np.bool_)):
            raise TypeError('`regex` must be a boolean')
        if not isinstance(keep, (bool, np.bool_)):
            raise TypeError('`keep` must be a boolean')

        if keep:
            columns, locs, other_columns, other_locs = self._validate_columns_others(column)
        else:
            columns, locs = self._validate_columns(column)

        data = self._df._data['O']
        if len(locs) == 1:
            arr = _sf.count(data[:, locs[0]], pat, case, flags, na, regex)[:, np.newaxis]
        else:
            arr = _sf.count_2d(data[:, locs], pat, case, flags, na, regex)

        if keep:
            return self._create_df_multiple_dtypes(arr, columns, locs, other_columns, other_locs)
        else:
            return self._create_df(arr, 'O', columns)

    def endswith(self, column=None, pat=None, keep=False):
        if not isinstance(pat, str):
            raise TypeError('`pat` must be a string')
        if not isinstance(keep, (bool, np.bool_)):
            raise TypeError('`keep` must be a boolean')

        if keep:
            columns, locs, other_columns, other_locs = self._validate_columns_others(column)
        else:
            columns, locs = self._validate_columns(column)

        data = self._df._data['O']
        if len(locs) == 1:
            arr = _sf.endswith(data[:, locs[0]], pat)[:, np.newaxis]
        else:
            arr = _sf.endswith_2d(data[:, locs], pat)

        if keep:
            return self._create_df_multiple_dtypes(arr, columns, locs, other_columns, other_locs)
        else:
            return self._create_df(arr, 'O', columns)
