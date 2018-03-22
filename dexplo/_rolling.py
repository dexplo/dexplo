from dexplo._frame import DataFrame
import dexplo._utils as utils
from collections import defaultdict, OrderedDict
import numpy as np
from numpy import ndarray
from typing import Union, Dict, List, Tuple, Callable
from dexplo._libs import rolling as _roll
import warnings
import weakref

ColInfoT = Dict[str, utils.Column]

def get_func_kwargs(name):
    if name in {'sum', 'prod', 'mean', 'median', 'size'}:
        return {}
    elif name in {'min', 'max'}:
        return dict(ignore_str=False, ignore_date=False)
    elif name in {'count'}:
        return dict(ignore_str=False, ignore_date=False, keep_date_type=False)
    elif name in {'first', 'last'}:
        return dict(ignore_str=False)
    elif name in {'any', 'all', 'nunique'}:
        return dict(ignore_str=False, ignore_date=False, keep_date_type=False)
    elif name in {'var'}:
        return dict(add_positions=True)


class Roller(object):

    def __init__(self, df: DataFrame, left: int, right: int, min_window: int,
                 kept_columns: Union[bool, str, List[str]]) -> None:
        self._df = weakref.ref(df)
        self._left = left
        self._right = right
        self._min_window = min_window
        self._kept_columns = self._get_kept_columns(kept_columns)

    def __repr__(self) -> str:
        return ("This is a rolling object. Here is some info on it:\n"
                f"The left is {self._left}\n"
                f"The right is {self._right - 1}")

    def __len__(self) -> int:
        return self._right - self._left

    def _get_kept_columns(self, kept_columns):
        if kept_columns is False:
            return []
        elif kept_columns is True:
            return self._df().columns
        else:
            if isinstance(kept_columns, str):
                kept_columns = [kept_columns]
            self._df()._validate_column_name_list(kept_columns)
            return kept_columns

    def _roll_generic(self, name, columns):
        if columns is None:
            columns = self._df().columns
        elif isinstance(columns, str):
            columns = [columns]
        elif not isinstance(columns, list):
            raise TypeError('`columns` must either be a string, a list of column names, or None')

        col_order = dict(zip(columns, range(len(columns))))

        dtype_locs = defaultdict(list)
        dtype_cols = defaultdict(list)
        col_info = self._df()._column_info
        for i, col in enumerate(columns):
            try:
                dtype, loc, order = col_info[col].values
            except KeyError:
                raise KeyError(f'{col} is not a column name')

            dtype_locs[dtype].append(loc)
            dtype_cols[dtype].append(col)

        kept_dtype_loc = defaultdict(list)
        new_col_info = {}
        dtype_ct = defaultdict(int)
        for i, col in enumerate(self._kept_columns):
            dtype, loc, _ = col_info[col].values
            new_loc = len(kept_dtype_loc[dtype])
            kept_dtype_loc[dtype].append(loc)
            new_col_info[col] = utils.Column(dtype, new_loc, i)
            dtype_ct[dtype] += 1

        data_dict = defaultdict(list)
        for dtype, locs in dtype_locs.items():
            func_name = name + '_' + utils.convert_kind_to_dtype_generic(dtype)
            data = self._df()._data[dtype]
            result = getattr(_roll, func_name)(data, np.array(locs),
                                               self._left, self._right, self._min_window)
            result_dtype = result.dtype.kind
            data_dict[result_dtype].append(result)
            for col in dtype_cols[dtype]:
                order = col_order[col]
                new_col = col
                if col in self._kept_columns:
                    new_col = col + '_rolling'
                    columns[columns.index(col)] = new_col
                new_col_info[new_col] = utils.Column(result_dtype, dtype_ct[result_dtype],
                                                     order + len(self._kept_columns))
                dtype_ct[result_dtype] += 1

        new_data = {}
        for dtype, locs in kept_dtype_loc.items():
            data = self._df()._data[dtype][:, locs]
            if data.ndim == 1:
                data = data[:, np.newaxis]
            new_data[dtype] = data

        for dtype, data in data_dict.items():
            if dtype not in new_data:
                new_data[dtype] = np.column_stack((*data,))
            else:
                new_data[dtype] = np.column_stack((new_data[dtype], *data))

        new_columns = np.concatenate((self._kept_columns, columns))
        return DataFrame._construct_from_new(new_data, new_col_info, new_columns)

    def count(self, columns=None) -> DataFrame:
        return self._roll_generic('count', columns)

    def sum(self, columns=None) -> DataFrame:
        return self._roll_generic('sum', columns)

    def prod(self, columns=None) -> DataFrame:
        return self._roll_generic('prod', columns)

    def mean(self, columns=None) -> DataFrame:
        return self._roll_generic('mean', columns)

    def max(self, columns=None) -> DataFrame:
        return self._roll_generic('max', columns)

    def min(self, columns=None) -> DataFrame:
        return self._roll_generic('min', columns)

    def median(self, columns=None) -> DataFrame:
        return self._roll_generic('median', columns)

    def var(self, ddof=1) -> DataFrame:
        return self._group_agg('var', add_positions=True, ddof=ddof)

    def cov(self) -> DataFrame:
        return self._cov_corr('cov')

    def corr(self) -> DataFrame:
        return self._cov_corr('corr')

    def _cov_corr(self, name: str) -> DataFrame:
        calc_columns: List[str] = []
        calc_dtype_loc: List[Tuple[str, int]] = []
        np_dtype = 'int64'
        for col in self._df()._columns:
            if col in self._group_columns:
                continue
            dtype, loc, order = self._df()._column_info[col].values
            if dtype in 'fib':
                if dtype == 'f':
                    np_dtype = 'float64'
                calc_columns.append(col)
                calc_dtype_loc.append((dtype, loc))

        data = self._df()._values_number_drop(calc_columns, calc_dtype_loc, np_dtype)
        dtype_word = utils.convert_kind_to_dtype(data.dtype.kind)
        func = getattr(_gb, name + '_' + dtype_word)
        result = func(self._group_labels, len(self), data, [])

        data_dict = self._get_group_col_data()
        data_dict_final: Dict[str, List[ndarray]] = defaultdict(list)
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

    def any(self) -> DataFrame:
        return self._group_agg('any', False, ignore_date=False, keep_date_type=False)

    def all(self) -> DataFrame:
        return self._group_agg('all', False, ignore_date=False, keep_date_type=False)



    def nunique(self) -> DataFrame:
        return self._group_agg('nunique', False, ignore_date=False, keep_date_type=False)


    def agg(self, *args):
        func_cols = OrderedDict()
        func_new_names = OrderedDict()
        func_order = OrderedDict()
        func_kwargs = OrderedDict()

        for i, arg in enumerate(args):
            if not isinstance(arg, tuple):
                raise TypeError('`Each argument to `agg` must be a 3-item tuple consisting '
                                'of aggregation function (name or callable), '
                                'aggregating column and resulting column name')
            if len(arg) not in (3, 4):
                raise ValueError(f'The tuple {arg} must have either three or four values '
                                 '- the aggregation function, aggregating column, and resulting '
                                 'column name. Optionally, it may have a dictionary of kwargs '
                                 'for its fourth element')
            if isinstance(arg[0], str):
                if arg[0] not in {'size', 'count', 'sum', 'prod', 'mean',
                                  'max', 'min', 'first', 'last', 'var',
                                  'cov', 'corr', 'any', 'all', 'median',
                                  'nunique'}:
                    raise ValueError(f'{arg[0]} is not a possible aggregation function')
            elif not isinstance(arg[0], Callable):
                raise TypeError('The first item of the tuple must be an aggregating function name '
                                'as a string or a user-defined function')

            if not isinstance(arg[1], str):
                raise TypeError('The second element in each tuple must be a column name as a '
                                'string.')
            elif arg[1] not in self._df()._column_info:
                raise ValueError(f'`{arg[1]}` is not a column name')

            if not isinstance(arg[2], str):
                raise TypeError('The third element in each tuple must be the name of the new '
                                'column as a string')

            if len(arg) == 4 and not isinstance(arg[3], dict):
                raise TypeError('The fourth element in the tuple must be a dictionary of '
                                'kwargs')

            func_name = arg[0]
            if arg[0] not in func_cols:
                func_cols[func_name] = [arg[1]]
                func_new_names[func_name] = [arg[2]]
                func_order[func_name] = [i]
                if len(arg) == 4:
                    func_kwargs[func_name] = [arg[3]] # a list of dictionaries for kwargs
                else:
                    func_kwargs[func_name] = [None]
            else:
                func_cols[func_name].append(arg[1])
                func_new_names[func_name].append(arg[2])
                func_order[func_name].append(i)
                if len(arg) == 4:
                    func_kwargs[func_name].append(arg[3])
                else:
                    func_kwargs[func_name].append(None)

        return self._single_agg(agg_cols=func_cols, new_names=func_new_names,
                                new_order=func_order, num_agg_cols=i + 1,
                                func_kwargs=func_kwargs)

    def filter(self, func, *args, **kwargs):
        if not isinstance(func, Callable):
            raise TypeError('The `func` varialbe must a function or any callable object')
        labels = self._group_labels
        size = len(self._group_position)
        result = _gb.filter(labels, size, self._df(), func, *args, **kwargs)

        new_data = {kind: data[result] for kind, data in self._df()._data.items()}
        new_col_info = self._df()._copy_column_info()
        columns = self._df()._columns.copy()

        return self._df()._construct_from_new(new_data, new_col_info, columns)

    def apply(self, func, *args, **kwargs):
        if not isinstance(func, Callable):
            raise TypeError('The `func` variable must be a function or any callable object')
        labels = self._group_labels
        size = len(self._group_position)
        new_data, new_column_info, new_columns, group_repeats = _gb.apply(labels, size, self._df(), func, *args, **kwargs)

        grouped_data_dict = self._get_group_col_data()
        grouped_column_info = self._get_new_column_info()
        grouped_columns = self._group_columns.copy()
        order_add = len(grouped_columns)

        new_column_info_final = {}
        for col in new_columns:
            dtype, loc, order = new_column_info[col].values
            loc_add = grouped_data_dict.get(dtype, 0)
            if loc_add != 0:
                loc_add = loc_add[0].shape[1]
            new_column_info_final[col] = utils.Column(dtype, loc + loc_add, order + order_add)

        new_grouped_columns = []
        for col in grouped_columns:
            if col in new_column_info_final:
                new_grouped_columns.append(col + '_group')
            else:
                new_grouped_columns.append(col)

        dtype_loc = defaultdict(int)
        for i, col in enumerate(grouped_columns):
            dtype = grouped_column_info[col].dtype
            loc = dtype_loc[dtype]
            new_col = new_grouped_columns[i]
            new_column_info_final[new_col] = utils.Column(dtype, loc, i)
            dtype_loc[dtype] += 1

        new_columns = np.concatenate((new_grouped_columns, new_columns))

        for dtype, data_list in grouped_data_dict.items():
            data = np.concatenate(data_list, 1)
            data = np.repeat(data, group_repeats, axis=0)
            if dtype not in new_data:
                new_data[dtype] = data
            else:
                new_data[dtype] = np.concatenate((data, new_data[dtype]), 1)

        return DataFrame._construct_from_new(new_data, new_column_info_final, new_columns)
