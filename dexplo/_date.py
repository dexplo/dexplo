import dexplo._utils as utils
from dexplo._libs import (string_funcs as _sf,
                          date as _date,
                          timedelta as _td)
import numpy as np
from numpy import nan, ndarray
from typing import (Union, Dict, List, Optional, Tuple, Callable, overload,
                    NoReturn, Set, Iterable, Any, TypeVar, Type, Generator)
import weakref
import datetime

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
            arr = getattr(self, name)(data[:, locs], **kwargs)
        else:
            arr = getattr(self, name)(data[:, locs], **kwargs)

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
        self._df = weakref.ref(df)()
        self._dtype_acc = 'M'
        self._2d = ''

    def ceil(self, column=None, freq=None, keep=False):
        if freq not in ('ns', 'us', 'ms', 's', 'm', 'h', 'D', 'Y'):
            raise ValueError("`freq` must be one of 'ns', 'us', 'ms', 's', 'm', 'h', 'D', 'Y'")
        return self._generic(name='_ceil', column=column, keep=keep, multiple=False, freq=freq)

    def _ceil(self, data, freq):
        if freq == 'ns':
            return data
        if freq == 'Y':
            if data.size < 5000:
                years = data.astype('datetime64[Y]')
                diff = (years - data).astype('int64')
                years[diff != 0] += 1
                return years.astype('datetime64[ns]')
        return getattr(_date,  'ceil_' + freq)(data.astype('float64'))

    def day(self, column=None, keep=False):
        return self._generic(name='_day', column=column, keep=keep, multiple=True)

    def _day(self, data):
        if data.size < 5000:
            days = (data.astype('datetime64[D]') - data.astype('datetime64[M]') + 1).astype('float64')
            days[np.isnat(data)] = nan
        else:
            return _date.day(data.astype('int64'))
        return days

    def day_of_week(self, column=None, keep=False):
        return self._generic(name='_day_of_week', column=column, keep=keep, multiple=True)

    def _day_of_week(self, data):
        return _date.day_of_week(data.astype('int64'))

    def day_of_year(self, column=None, keep=False):
        return self._generic(name='_day_of_year', column=column, keep=keep, multiple=True)

    def _day_of_year(self, data):
        if data.size < 2500:
            doy = (data.astype('datetime64[D]') - data.astype('datetime64[Y]') + 1).astype('float64')
            doy[np.isnat(data)] = nan
            return doy
        else:
            return _date.day_of_year(data.astype('int64'))

    def days_in_month(self, column=None, keep=False):
        return self._generic(name='_days_in_month', column=column, keep=keep, multiple=True)

    def _days_in_month(self, data):
        if data.size < 5000:
            month = data.astype('datetime64[M]')
            next_month = (month + 1).astype('datetime64[D]')
            dim = (next_month - month).astype('float64')
            dim[np.isnat(data)] = nan
            return dim
        else:
            return _date.days_in_month(data.astype('int64'))

    def floor(self, column=None, freq=None, keep=False):
        if freq not in ('ns', 'us', 'ms', 's', 'm', 'h', 'D', 'Y'):
            raise ValueError("`freq` must be one of 'ns', 'us', 'ms', 's', 'm', 'h', 'D', 'Y'")
        return self._generic(name='_floor', column=column, keep=keep, multiple=False, freq=freq)

    def _floor(self, data, freq):
        if freq == 'ns':
            return data
        return getattr(_date,  'floor_' + freq)(data.astype('int64'))

    def hour(self, column=None, keep=False):
        return self._generic(name='_hour', column=column, keep=keep, multiple=True)

    def _hour(self, data):
        return _date.hour(data.astype('int64'))

    def is_leap_year(self, column=None, keep=False):
        return self._generic(name='_is_leap_year', column=column, keep=keep, multiple=True)

    def _is_leap_year(self, data):
        if data.size < 500:
            years = data.astype('datetime64[Y]').astype('float64') + 1970
            years[np.isnat(data)] = nan
            return np.where(years % 4 == 0, np.where(years % 100 == 0,
                                              np.where(years % 400 == 0, True, False), True), False)
        else:
            return _date.is_leap_year(data.astype('int64'))

    def is_month_end(self, column=None, keep=False):
        return self._generic(name='_is_month_end', column=column, keep=keep, multiple=True)

    def _is_month_end(self, data):
        return self._day(data) == self._days_in_month(data)

    def is_month_start(self, column=None, keep=False):
        return self._generic(name='_is_month_start', column=column, keep=keep, multiple=True)

    def _is_month_start(self, data):
        return self._day(data) == 1

    def is_quarter_end(self, column=None, keep=False):
        return self._generic(name='_is_quarter_end', column=column, keep=keep, multiple=True)

    def _is_quarter_end(self, data):
        return _date.is_quarter_end(data.astype('int64'))

    def is_quarter_start(self, column=None, keep=False):
        return self._generic(name='_is_quarter_start', column=column, keep=keep, multiple=True)

    def _is_quarter_start(self, data):
        return _date.is_quarter_start(data.astype('int64'))

    def is_year_end(self, column=None, keep=False):
        return self._generic(name='_is_year_end', column=column, keep=keep, multiple=True)

    def _is_year_end(self, data):
        return _date.is_year_end(data.astype('int64'))

    def is_year_start(self, column=None, keep=False):
        return self._generic(name='_is_year_start', column=column, keep=keep, multiple=True)

    def _is_year_start(self, data):
        return _date.is_year_start(data.astype('int64'))

    def minute(self, column=None, keep=False):
        return self._generic(name='_minute', column=column, keep=keep, multiple=True)

    def _minute(self, data):
        return _date.minute(data.astype('int64'))

    def month(self, column=None, keep=False):
        return self._generic(name='_month', column=column, keep=keep, multiple=True)

    def _month(self, data):
        if data.size < 6000:
            return _date.month(data.astype('datetime64[M]').astype('int64'))
        else:
            return _date.month2(data.astype('int64'))

    def quarter(self, column=None, keep=False):
        return self._generic(name='_quarter', column=column, keep=keep, multiple=True)

    def _quarter(self, data):
        if data.size < 5000:
            t = data.astype('datetime64[M]').astype('float64') % 12 // 3 + 1
            t[np.isnat(data)] = nan
            return t
        else:
            return _date.quarter(data.astype('int64'))

    def second(self, column=None, keep=False):
        return self._generic(name='_second', column=column, keep=keep, multiple=True)

    def _second(self, data):
        return _date.second(data.astype('int64'))

    def millisecond(self, column=None, keep=False):
        return self._generic(name='_millisecond', column=column, keep=keep, multiple=True)

    def _millisecond(self, data):
        return _date.millisecond(data.astype('int64'))

    def microsecond(self, column=None, keep=False):
        return self._generic(name='_microsecond', column=column, keep=keep, multiple=True)

    def _microsecond(self, data):
        return _date.microsecond(data.astype('int64'))

    def nanosecond(self, column=None, keep=False):
        return self._generic(name='_nanosecond', column=column, keep=keep, multiple=True)

    def _nanosecond(self, data):
        return _date.nanosecond(data.astype('int64'))

    def round(self, column=None, freq=None, keep=False):
        if freq not in ('ns', 'us', 'ms', 's', 'm', 'h', 'D', 'Y'):
            raise ValueError("`freq` must be one of 'ns', 'us', 'ms', 's', 'm', 'h', 'D', 'Y'")
        return self._generic(name='_round', column=column, keep=keep, multiple=False, freq=freq)

    def _round(self, data, freq):
        if freq == 'ns':
            return data
        return getattr(_date,  'round_' + freq)(data.astype('float64'))

    def strftime(self, column=None, date_format=None, keep=False):
        if not isinstance(date_format, str):
            raise TypeError('`date_format` must be a str')
        return self._generic(name='_strftime', column=column, keep=keep, multiple=True,
                             date_format=date_format)

    def _strftime(self, data, date_format):
        return _date.strftime(data.astype('float64'), date_format)

    def to_pytime(self, column=None):
        columns, locs = self._validate_columns(column)
        data = self._df._data['M'][:, locs]
        return _date.to_pytime(data.astype('float64'))

    def to_pydatetime(self, column=None):
        columns, locs = self._validate_columns(column)
        data = self._df._data['M'][:, locs]
        # return _date.to_pydatetime(data.astype('int64'))
        return data.astype('datetime64[us]').astype(datetime.datetime)


    def week_of_year(self, column=None, keep=False):
        return self._generic(name='_week_of_year', column=column, keep=keep, multiple=True)

    def _week_of_year(self, data):
        if data.size < 500:
            woy = (self._day_of_year(data) - self._day_of_week(data) + 9) // 7
            years = self._year(data)
            return _date.week_of_year(woy, years)
        else:
            return _date.week_of_year2(data.astype('int64'))

    def weekday_name(self, column=None, keep=False):
        return self._generic(name='_weekday_name', column=column, keep=keep, multiple=True)

    def _weekday_name(self, data):
        dow = data.astype('int64')
        return _date.weekday_name(dow)

    def year(self, column=None, keep=False):
        return self._generic(name='_year', column=column, keep=keep, multiple=True)

    def _year(self, data):
        if data.size < 5000:
            years = data.astype('datetime64[Y]').astype('float64') + 1970
            years[np.isnat(data)] = nan
            return years
        else:
            return _date.year(data.astype('int64'))


class TimeDeltaClass(AccessorMixin):

    def __init__(self, df):
        self._df = weakref.ref(df)()
        self._dtype_acc = 'm'
        self._2d = ''

    def seconds(self, column=None, total=False, keep=False):
        if not isinstance(total, (bool, np.bool_)):
            raise TypeError('`total` must be a boolean')
        return self._generic(name='_seconds', column=column, keep=keep, multiple=True,
                             total=total)

    def _seconds(self, data, total):
        if total:
            t = data.astype('float64') / 10 ** 9
            t[np.isnat(data)] = nan
            return t
        else:
            return _td.seconds(data.astype('int64'))

    def milliseconds(self, column=None, total=False, keep=False):
        if not isinstance(total, (bool, np.bool_)):
            raise TypeError('`total` must be a boolean')
        return self._generic(name='_milliseconds', column=column, keep=keep, multiple=True,
                             total=total)

    def _milliseconds(self, data, total):
        return _td.milliseconds(data.astype('int64'), total)

    def microseconds(self, column=None, total=False, keep=False):
        if not isinstance(total, (bool, np.bool_)):
            raise TypeError('`total` must be a boolean')
        return self._generic(name='_microseconds', column=column, keep=keep, multiple=True,
                             total=total)

    def _microseconds(self, data, total):
        return _td.microseconds(data.astype('int64'), total)

    def nanoseconds(self, column=None, keep=False):
        return self._generic(name='_nanoseconds', column=column, keep=keep, multiple=True)

    def _nanoseconds(self, data):
        return _td.nanoseconds(data.astype('int64'))

    def days(self, column=None, total=False, keep=False):
        if not isinstance(total, (bool, np.bool_)):
            raise TypeError('`total` must be a boolean')
        return self._generic(name='_days', column=column, keep=keep, multiple=True,
                             total=total)

    def _days(self, data, total):
        return _td.days(data.astype('int64'), total)


