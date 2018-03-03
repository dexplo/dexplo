import dexplo._utils as utils
from dexplo._libs import (string_funcs as _sf,
                          math as _math)
import re
import numpy as np
from numpy import nan, ndarray
from typing import (Union, Dict, List, Optional, Tuple, Callable, overload,
                    NoReturn, Set, Iterable, Any, TypeVar, Type, Generator)
from typing import Pattern
import textwrap
import weakref


class StringClass(object):

    def __init__(self, df: 'DataFrame') -> None:
        self._df = weakref.ref(df)()
        self._dtype_acc = 'O'
        self._2d = '_2d'

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
            for old_dtype, old_data in self._df._data.items():
                if old_dtype == 'O':
                    new_data['O'] = arr
                else:
                    new_data[old_dtype] = old_data.copy('F')
        else:
            new_data = {}
            add_loc = 0
            if dtype in self._df._data:
                add_loc = self._df._data[dtype].shape[1]
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

    def _str_generic_concat(self, name, column, keep, return_dtype, **kwargs):
        if not isinstance(keep, (bool, np.bool_)):
            raise TypeError('`keep` must be a boolean')

        if column is None:
            columns = []
            locs = []
            for col in self._df._columns:
                dtype, loc, _ = self._df._column_info[col].values
                if dtype == 'O':
                    columns.append(col)
                    locs.append(loc)
        else:
            columns, locs = self._validate_columns(column)

        data = self._df._data['O']

        count = 0
        if return_dtype != 'O':
            if return_dtype in self._df._data:
                count = self._df._data[return_dtype].shape[1]
        else:
            count = self._df._data['O'].shape[1] - len(columns)

        kwargs['count'] = count

        final_arr, final_cols, group_len = getattr(_sf, name)(data[:, locs], **kwargs)
        dtype_new = final_arr.dtype.kind

        if len(columns) > 1:
            final_cols = np.repeat(columns, group_len).astype('O') + '_' + final_cols
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
                    for i in range(arr.shape[1]):
                        final_arr[:, i] = arr[:, i]
                    new_data[dtype_new] = final_arr
                else:
                    new_data[dtype] = arr

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

    def _str_generic(self, name, column, keep, multiple, **kwargs):
        if not isinstance(keep, (bool, np.bool_)):
            raise TypeError('`keep` must be a boolean')

        if keep:
            columns, locs, other_columns, other_locs = self._validate_columns_others(column)
        else:
            columns, locs = self._validate_columns(column)

        data = self._df._data['O']
        if len(locs) == 1:
            arr = getattr(_sf, name)(data[:, locs[0]], **kwargs)[:, np.newaxis]
        else:
            arr = getattr(_sf, name + '_2d')(data[:, locs], **kwargs)

        if keep:
            if multiple:
                return self._create_df_multiple_dtypes(arr, columns, locs, other_columns, other_locs)
            else:
                data = data.copy()
                for i, loc in enumerate(locs):
                    data[:, loc] = arr[:, i]
                return self._create_df_all(data, 'O')
        else:
            return self._create_df(arr, arr.dtype.kind, columns)

    def capitalize(self, column=None, keep=False):
        return self._str_generic(name='capitalize', column=column, keep=keep, multiple=False)

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
            # return DataFrame(data)

    def center(self, column=None, width=None, fillchar=' ', keep=False):
        if not isinstance(fillchar, str):
            raise TypeError('`fillchar` must be a string')
        elif len(fillchar) != 1:
            raise ValueError('`fillchar` must be exactly one character long')
        if not isinstance(width, (int, np.integer)):
            raise TypeError('`width` must be an integer')

        return self._str_generic(name='center', column=column, keep=keep, multiple=False,
                                 width=width, fillchar=fillchar)

    def contains(self, column=None, pat=None, case=True, flags=0, na=nan, regex=True, keep=False):
        if not isinstance(case, (bool, np.bool_)):
            raise TypeError('`case` must be a boolean')
        if not isinstance(flags, (int, np.integer, re.RegexFlag)):
            raise TypeError('flags must be a `RegexFlag` or integer')
        if not isinstance(pat, (str, Pattern)):
            raise TypeError('`pat` must either be either a string or compiled regex pattern')
        if not isinstance(regex, (bool, np.bool_)):
            raise TypeError('`regex` must be a boolean')

        return self._str_generic(name='contains', column=column, keep=keep, multiple=True,
                                 pat=pat, case=case, flags=flags, na=na, regex=regex)

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

        return self._str_generic(name='count', column=column, keep=keep, multiple=True,
                                 pat=pat, case=case, flags=flags, na=na, regex=regex)

    def endswith(self, column=None, pat=None, keep=False):
        if not isinstance(pat, str):
            raise TypeError('`pat` must be a string')

        return self._str_generic(name='endswith', column=column, keep=keep, multiple=True,
                                 pat=pat)

    def find(self, column=None, sub=None, start=None, end=None, keep=False):
        if not isinstance(sub, str):
            raise TypeError('`sub` must be a string')
        if start is not None and not isinstance(start, (int, np.integer)):
            raise TypeError('`start` must be an intege or None')
        if end is not None and not isinstance(start, (int, np.integer)):
            raise TypeError('`end` must be an integer or None')

        return self._str_generic(name='find', column=column, keep=keep, multiple=True,
                                 sub=sub, start=start, end=end)

    def findall(self, column=None, pat=None, pos=0, case=True, flags=0, keep=False):
        if not isinstance(pat, (str, Pattern)):
            raise TypeError('`pat` must be a str or compiled regular expression')
        if not isinstance(pos, (int, np.integer)):
            raise TypeError('`n` must be an integer')
        if not isinstance(case, (bool, np.bool_)):
            raise TypeError('`case` must be a boolean')
        if not isinstance(flags, (int, np.integer, re.RegexFlag)):
            raise TypeError('flags must be a `RegexFlag` or integer')

        return self._str_generic_concat('findall', column, keep, pat=pat, pos=pos,
                                        case=case, flags=flags, return_dtype='O')

    def get(self, column=None, i=None, keep=False):
        if not isinstance(i, (int, np.integer)):
            raise TypeError('`i` must be an intege or None')

        return self._str_generic(name='get', column=column, keep=keep, multiple=False,
                                 i=i)

    def get_dummies(self, column=None, sep=None, keep=False):
        if not isinstance(sep, str) and sep is not None:
            raise TypeError('`sep` must be an integer or None')

        return self._str_generic_concat('get_dummies', column, keep, sep=sep, return_dtype='i')

    def isalnum(self, column=None, keep=False):
        return self._str_generic(name='isalnum', column=column, keep=keep, multiple=True)

    def isalpha(self, column=None, keep=False):
        return self._str_generic(name='isalpha', column=column, keep=keep, multiple=True)

    def isdecimal(self, column=None, keep=False):
        return self._str_generic(name='isdecimal', column=column, keep=keep, multiple=True)

    def isdigit(self, column=None, keep=False):
        return self._str_generic(name='isdigit', column=column, keep=keep, multiple=True)

    def islower(self, column=None, keep=False):
        return self._str_generic(name='islower', column=column, keep=keep, multiple=True)

    def isnumeric(self, column=None, keep=False):
        return self._str_generic(name='isnumeric', column=column, keep=keep, multiple=True)

    def isspace(self, column=None, keep=False):
        return self._str_generic(name='isspace', column=column, keep=keep, multiple=True)

    def istitle(self, column=None, keep=False):
        return self._str_generic(name='istitle', column=column, keep=keep, multiple=True)

    def isupper(self, column=None, keep=False):
        return self._str_generic(name='isupper', column=column, keep=keep, multiple=True)

    def join(self, column=None, sep=None, keep=False):
        if not isinstance(sep, str):
            raise TypeError('`sep` must be a string')
        return self._str_generic(name='join', column=column, keep=keep, multiple=False,
                                 sep=sep)

    def len(self, column=None, keep=False):
        return self._str_generic(name='_len', column=column, keep=keep, multiple=True)

    def ljust(self, column=None, width=None, fillchar=' ', keep=False):
        if not isinstance(fillchar, str):
            raise TypeError('`fillchar` must be a string')
        elif len(fillchar) != 1:
            raise ValueError('`fillchar` must be exactly one character long')
        if not isinstance(width, (int, np.integer)):
            raise TypeError('`width` must be an integer')

        return self._str_generic(name='ljust', column=column, keep=keep, multiple=False,
                                 width=width, fillchar=fillchar)

    def lower(self, column=None, keep=False):
        return self._str_generic(name='lower', column=column, keep=keep, multiple=False)

    def lstrip(self, column=None, to_strip=None, keep=False):
        if not isinstance(to_strip, str) and to_strip is not None:
            raise TypeError('`to_strip` must be a str or None')
        return self._str_generic(name='lstrip', column=column, keep=keep, multiple=False,
                                 to_strip=to_strip)

    def repeat(self, column=None, repeats=None, keep=False):
        if not isinstance(repeats, (int, np.integer)):
            raise TypeError('`repeats` must be an intege or None')

        return self._str_generic(name='repeat', column=column, keep=keep, multiple=False,
                                 repeats=repeats)

    def partition(self, column=None, sep='', keep=False):
        if not isinstance(sep, str):
            raise TypeError('`sep` must be an intege or None')

        return self._str_generic_concat('partition', column, keep, sep=sep, return_dtype='O')

    def replace(self, column=None, pat=None, repl=None, n=0, case=True, flags=0, keep=False):
        if not isinstance(case, (bool, np.bool_)):
            raise TypeError('`case` must be a boolean')
        if not isinstance(flags, (int, np.integer, re.RegexFlag)):
            raise TypeError('`flags` must be a `RegexFlag` or integer')
        if not isinstance(n, (int, np.integer, re.RegexFlag)):
            raise TypeError('`n` must be a `RegexFlag` or integer')
        if not isinstance(pat, (str, Pattern)):
            raise TypeError('`pat` must either be either a string or compiled regex pattern')
        if not isinstance(repl, str) or callable(repl):
            raise TypeError('`repl` must either be either a string or compiled regex pattern')

        return self._str_generic(name='replace', column=column, keep=keep, multiple=False,
                                 pat=pat, repl=repl, n=n, case=case, flags=flags)

    def rfind(self, column=None, sub=None, start=None, end=None, keep=False):
        if not isinstance(sub, str):
            raise TypeError('`sub` must be a string')
        if start is not None and not isinstance(start, (int, np.integer)):
            raise TypeError('`start` must be an intege or None')
        if end is not None and not isinstance(start, (int, np.integer)):
            raise TypeError('`end` must be an integer or None')

        return self._str_generic(name='rfind', column=column, keep=keep, multiple=True,
                                 sub=sub, start=start, end=end)

    def rjust(self, column=None, width=None, fillchar=' ', keep=False):
        if not isinstance(fillchar, str):
            raise TypeError('`fillchar` must be a string')
        elif len(fillchar) != 1:
            raise ValueError('`fillchar` must be exactly one character long')
        if not isinstance(width, (int, np.integer)):
            raise TypeError('`width` must be an integer')

        return self._str_generic(name='rjust', column=column, keep=keep, multiple=False,
                                 width=width, fillchar=fillchar)

    def rpartition(self, column=None, sep='', keep=False):
        if not isinstance(sep, str):
            raise TypeError('`sep` must be an intege or None')

        return self._str_generic_concat('rpartition', column, keep, sep=sep, return_dtype='O')

    def rsplit(self, column=None, pat=None, n=0, case=True, flags=0, keep=False):
        if not isinstance(pat, (str, Pattern)):
            raise TypeError('`pat` must be a str or compiled regular expression')
        if not isinstance(n, (int, np.integer)):
            raise TypeError('`n` must be an integer')
        if not isinstance(case, (bool, np.bool_)):
            raise TypeError('`case` must be a boolean')
        if not isinstance(flags, (int, np.integer, re.RegexFlag)):
            raise TypeError('flags must be a `RegexFlag` or integer')

        return self._str_generic_concat('rsplit', column, keep, pat=pat, n=n, case=case, flags=flags,
                                        return_dtype='O')

    def rstrip(self, column=None, to_strip=None, keep=False):
        if not isinstance(to_strip, str) and to_strip is not None:
            raise TypeError('`to_strip` must be a str or None')
        return self._str_generic(name='rstrip', column=column, keep=keep, multiple=False,
                                 to_strip=to_strip)

    def slice(self, column=None, start=None, stop=None, step=None, keep=False):
        if not isinstance(start, (int, np.integer)) and start is not None:
            raise TypeError('`start` must be an integer')
        if not isinstance(stop, (int, np.integer)) and stop is not None:
            raise TypeError('`stop` must be an integer')
        if not isinstance(step, (int, np.integer)) and step is not None:
            raise TypeError('`step` must be an integer')
        return self._str_generic(name='_slice', column=column, keep=keep, multiple=False,
                                 start=start, stop=stop, step=step)

    def slice_replace(self, column=None, start=None, stop=None, repl=None, keep=False):
        if start is None:
            start = 0
        if not isinstance(start, (int, np.integer)):
            raise TypeError('`start` must be an integer')
        if not isinstance(stop, (int, np.integer)) and stop is not None:
            raise TypeError('`stop` must be an integer')
        if repl is None:
            repl = ''
        if not isinstance(repl, str):
            raise TypeError('`repl` must be a str or None')
        return self._str_generic(name='slice_replace', column=column, keep=keep, multiple=False,
                                 start=start, stop=stop, repl=repl)

    def split(self, column=None, pat=None, n=0, case=True, flags=0, keep=False):
        if not isinstance(pat, (str, Pattern)):
            raise TypeError('`pat` must be a str or compiled regular expression')
        if not isinstance(n, (int, np.integer)):
            raise TypeError('`n` must be an integer')
        if not isinstance(case, (bool, np.bool_)):
            raise TypeError('`case` must be a boolean')
        if not isinstance(flags, (int, np.integer, re.RegexFlag)):
            raise TypeError('flags must be a `RegexFlag` or integer')

        return self._str_generic_concat('split', column, keep, pat=pat, n=n, case=case, flags=flags,
                                        return_dtype='O')

    def startswith(self, column=None, pat=None, keep=False):
        if not isinstance(pat, str):
            raise TypeError('`pat` must be a string')

        return self._str_generic(name='startswith', column=column, keep=keep, multiple=True,
                                 pat=pat)

    def strip(self, column=None, to_strip=None, keep=False):
        if not isinstance(to_strip, str) and to_strip is not None:
            raise TypeError('`to_strip` must be a str or None')
        return self._str_generic(name='strip', column=column, keep=keep, multiple=False,
                                 to_strip=to_strip)

    def swapcase(self, column=None, keep=False):
        return self._str_generic(name='swapcase', column=column, keep=keep, multiple=False)

    def title(self, column=None, keep=False):
        return self._str_generic(name='title', column=column, keep=keep, multiple=False)

    def translate(self, column=None, table=None, keep=False):
        if not isinstance(table, dict):
            raise TypeError('`table` must be a dict')
        return self._str_generic(name='translate', column=column, keep=keep, multiple=False,
                                 table=table)

    def upper(self, column=None, keep=False):
        return self._str_generic(name='upper', column=column, keep=keep, multiple=False)

    def wrap(self, column=None, width=None, keep=False, **kwargs):
        if not isinstance(width, (int, np.integer)):
            raise TypeError('`width` must be an integer')

        t = textwrap.TextWrapper(width, **kwargs)

        return self._str_generic(name='wrap', column=column, keep=keep, multiple=False,
                                 t=t)

    def zfill(self, column=None, width=None, keep=False):
        if not isinstance(width, (int, np.integer)):
            raise TypeError('`width` must be an integer')

        return self._str_generic(name='zfill', column=column, keep=keep, multiple=False,
                                 width=width)

