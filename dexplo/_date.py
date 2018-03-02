from dexplo._frame import DataFrame
import dexplo._utils as utils
from dexplo._libs import (string_funcs as _sf,
                          math as _math)
import numpy as np
from numpy import nan, ndarray
from typing import (Union, Dict, List, Optional, Tuple, Callable, overload,
                    NoReturn, Set, Iterable, Any, TypeVar, Type, Generator)

NAT = np.datetime64('nat')


class AccessorMixin(object):

    def _validate_columns(self, column):
        if isinstance(column, str):
            try:
                dtype, loc, _ = self._df._column_info[column].values
            except KeyError:
                raise KeyError(f'Column "{column}" does not exist in the DataFrame')
            if dtype != self._dtype_acc:
                raise ValueError(f'Column name "{column}" is not '
                                 f'a {utils.convert_kind_to_dtype(self._dtype_acc)} column')
            return [column], [loc]
        elif isinstance(column, list):
            locs = []
            for col in column:
                try:
                    dtype, loc, _ = self._df._column_info[col].values
                except KeyError:
                    raise KeyError(f'Column {col} does not exist in the DataFrame')

                if dtype != 'O':
                    raise ValueError(f'Column name "{col}" is not a '
                                     f'{utils.convert_kind_to_dtype(self._dtype_acc)} column')
                locs.append(self._df._column_info[col].loc)
            if len(column) != len(set(column)):
                raise ValueError('You cannot complete this operation with duplicate columns')
            return column, locs
        elif column is None:
            locs = []
            columns = []
            for col in self._df._columns:
                dtype, loc, _ = self._df._column_info[col].values
                if dtype == self._dtype_acc:
                    columns.append(col)
                    locs.append(loc)
            return columns, locs
        else:
            raise TypeError('`column` must be a column name as a string, a list of string, or None')

    def _validate_columns_others(self, column):
        if isinstance(column, str):
            column = [column]

        str_cols = []
        str_locs = []
        for col in self._df._columns:
            dtype, loc, _ = self._df._column_info[col].values
            if dtype == self._dtype_acc:
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

                if dtype != self._dtype_acc:
                    raise ValueError(f'Column name "{col}" is not a'
                                     f' {utils.convert_kind_to_dtype(self._dtype_acc)} column')

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
        if dtype == self._dtype_acc:
            for old_dtype, old_data in self._df._data.items():
                if old_dtype == self._dtype_acc:
                    new_data[self._dtype_acc] = arr
                else:
                    new_data[old_dtype] = old_data.copy('F')
        else:
            new_data = {}
            add_loc = 0
            if dtype in self._data:
                add_loc = self._data[dtype].shape[1]
            for old_dtype, old_data in self._df._data.items():
                if dtype != self._dtype_acc:
                    new_data[old_dtype] = old_data.copy('F')

            if dtype in new_data:
                new_data[dtype] = np.asfortranarray(np.column_stack((new_data[dtype], arr)))
            else:
                new_data[dtype] = arr

            new_column_info = {}
            for col, col_obj in self._df._column_info.items():
                old_dtype, loc, order = col_obj.values
                if old_dtype == self._dtype_acc:
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
            if dtype == self._dtype_acc:
                new_data[self._dtype_acc] = arr[:, locs_other]
            elif dtype == dtype_new:
                new_data[dtype_new] = np.asfortranarray(np.column_stack((arr, arr_new)))
            else:
                new_data[dtype] = arr.copy('F')

        if dtype_new not in new_data:
            new_data[dtype_new] = arr_new

        new_column_info = {}
        for col, col_obj in self._df._column_info.items():
            old_dtype, loc, order = col_obj.values
            if old_dtype != self._dtype_acc:
                new_column_info[col] = utils.Column(old_dtype, loc, order)

        # str columns that have changed type
        for i, (col, loc) in enumerate(zip(columns, column_locs)):
            order = self._df._column_info[col].order
            new_column_info[col] = utils.Column(dtype_new, add_loc + i, order)

        # those that stayed self._dtype_acc
        for i, col in enumerate(columns_other):
            order = self._df._column_info[col].order
            new_column_info[col] = utils.Column(self._dtype_acc, i, order)

        return self._df._construct_from_new(new_data, new_column_info, self._df._columns.copy())

    def _get_other_str_cols(self, columns):
        col_set = set(columns)
        cols_other = []
        locs_other = []
        for col in self._df._columns:
            if col not in col_set and self._df._column_info[col].dtype == self._dtype_acc:
                cols_other.append(col)
                locs_other.append(self._df._column_info[col].loc)
        return cols_other, locs_other

    def _generic_concat(self, name, column, keep, **kwargs):
        if not isinstance(keep, (bool, np.bool_)):
            raise TypeError('`keep` must be a boolean')

        if column is None:
            columns = []
            locs = []
            for col in self._df._columns:
                dtype, loc, _ = self._df._column_info[col].values
                if dtype == self._dtype_acc:
                    columns.append(col)
                    locs.append(loc)
        else:
            columns, locs = self._validate_columns(column)

        data = self._df._data[self._dtype_acc]
        arrs = []
        all_cols = []
        for loc in locs:
            arr, new_columns = getattr(_sf, name)(data[:, loc], **kwargs)
            arrs.append(arr)
            all_cols.append(new_columns)

        dtype_new = arrs[0].dtype.kind

        if len(arrs) == 1:
            final_arr = arrs[0]
            final_cols = all_cols[0]
        else:
            final_arr = np.column_stack(arrs)
            all_cols_new = []
            for cols, orig_name in zip(all_cols, columns):
                all_cols_new.append(cols + '_' + orig_name)
            final_cols = np.concatenate(all_cols_new)

        new_column_info = {}
        new_data = {}
        add_loc = 0
        add_order = 0
        if keep:
            df = self._df.drop(columns=columns)
            if dtype_new in df._data:
                add_loc = df._data[dtype_new].shape[1]
            add_order = df.shape[1]

            for dtype, arr in df._data.items():
                if dtype == dtype_new:
                    new_data[dtype_new] = np.column_stack((arr, final_arr))
                else:
                    new_data[dtype] = arr.copy('F')

            if dtype_new not in df._data:
                new_data[dtype_new] = final_arr

            new_column_info = df._copy_column_info()
            new_columns = np.concatenate((df._columns, final_cols))
        else:
            new_data = {dtype_new: final_arr}
            new_columns = final_cols

        for i, col in enumerate(final_cols):
            new_column_info[col] = utils.Column(dtype_new, i + add_loc, i + add_order)

        return self._df._construct_from_new(new_data, new_column_info, new_columns)

    def _generic(self, name, column, keep, multiple, **kwargs):
        if not isinstance(keep, (bool, np.bool_)):
            raise TypeError('`keep` must be a boolean')

        if keep:
            columns, locs, other_columns, other_locs = self._validate_columns_others(column)
        else:
            columns, locs = self._validate_columns(column)

        data = self._df._data[self._dtype_acc]
        if len(locs) == 1:
            arr = getattr(self, name)(data[:, locs[0]], **kwargs)[:, np.newaxis]
        else:
            arr = getattr(self, name + self._2d)(data[:, locs], **kwargs)

        if keep:
            if multiple:
                return self._create_df_multiple_dtypes(arr, columns, locs, other_columns, other_locs)
            else:
                data = data.copy()
                for i, loc in enumerate(locs):
                    data[:, loc] = arr[:, i]
                return self._create_df_all(data, self._dtype_acc)
        else:
            return self._create_df(arr, arr.dtype.kind, columns)


class DateTimeClass(AccessorMixin):

    def __init__(self, df):
        self._df = df
        self._dtype_acc = 'M'
        self._2d = ''

    def year(self, column=None, keep=False):
        return self._generic(name='_year', column=column, keep=keep, multiple=True)

    def _year(self, data):
        years = data.astype('datetime64[Y]').astype('float64') + 1970
        years[np.isnat(data)] = nan
        return years

    def month(self, column=None, keep=False):
        return self._generic(name='_month', column=column, keep=keep, multiple=True)

    def _month(self, data):
        months = data.astype('datetime64[M]').astype('float64') % 12 + 1
        months[np.isnat(data)] = nan
        return months

    def day(self, column=None, keep=False):
        return self._generic(name='_day', column=column, keep=keep, multiple=True)

    def _day(self, data):
        days = (data.astype('datetime64[D]') - data.astype('datetime64[M]') + 1).astype('float64')
        days[np.isnat(data)] = nan
        return days

    def hour(self, column=None, keep=False):
        return self._generic(name='_hour', column=column, keep=keep, multiple=True)

    def _hour(self, data):
        hours = (data.astype('datetime64[h]') - data.astype('datetime64[D]')).astype('float64')
        hours[np.isnat(data)] = nan
        return hours

    def minute(self, column=None, keep=False):
        return self._generic(name='_minute', column=column, keep=keep, multiple=True)

    def _minute(self, data):
        minutes = (data.astype('datetime64[m]') - data.astype('datetime64[h]')).astype('float64')
        minutes[np.isnat(data)] = nan
        return minutes

    def second(self, column=None, keep=False):
        return self._generic(name='_second', column=column, keep=keep, multiple=True)

    def _second(self, data):
        t = (data.astype('datetime64[s]') - data.astype('datetime64[m]')).astype('float64')
        t[np.isnat(data)] = nan
        return t

    def millisecond(self, column=None, keep=False):
        return self._generic(name='_millisecond', column=column, keep=keep, multiple=True)

    def _millisecond(self, data):
        t = (data.astype('datetime64[ms]') - data.astype('datetime64[s]')).astype('float64')
        t[np.isnat(data)] = nan
        return t

    def microsecond(self, column=None, keep=False):
        return self._generic(name='_microsecond', column=column, keep=keep, multiple=True)

    def _microsecond(self, data):
        t = (data.astype('datetime64[us]') - data.astype('datetime64[ms]')).astype('float64')
        t[np.isnat(data)] = nan
        return t

    def nanosecond(self, column=None, keep=False):
        return self._generic(name='_nanosecond', column=column, keep=keep, multiple=True)

    def _nanosecond(self, data):
        t = (data.astype('datetime64[ns]') - data.astype('datetime64[us]')).astype('float64')
        t[np.isnat(data)] = nan
        return t


class TimeDeltaClass(AccessorMixin):

    def __init__(self, df):
        self._df = df
        self._dtype_acc = 'm'
        self._2d = ''

