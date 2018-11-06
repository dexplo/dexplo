from dexplo._frame import DataFrame
import dexplo._utils as utils
from collections import defaultdict, OrderedDict
import numpy as np
from numpy import ndarray
from typing import Union, Dict, List, Tuple, Callable
from dexplo._libs import groupby as _gb
import warnings

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


class Grouper(object):

    def __init__(self, df: DataFrame, columns: List[str]) -> None:
        self._df = df
        self._group_labels, self._group_position = self._create_groups(columns)
        self._group_columns = columns

        if len(self._group_position) == self._df.shape[0]:
            warnings.warn("Each group contains exactly one row of data. "
                          "Are you sure you are grouping correctly?")

    def _create_groups(self, columns: Union[str, List[str]]) -> Tuple[ndarray, ndarray]:
        self._group_dtype_loc: Dict[str, List[int]] = defaultdict(list)
        self._column_info: ColInfoT = {}
        for i, col in enumerate(columns):
            dtype, loc, _ = self._df._column_info[col].values  # type: str, int, int
            cur_loc = len(self._group_dtype_loc[dtype])
            self._group_dtype_loc[dtype].append(loc)
            self._column_info[col] = utils.Column(dtype, cur_loc, i)

        if len(columns) == 1:
            # since there is just one column, dtype is from the for-loop
            final_arr = self._df._data[dtype][:, loc]
            if dtype in 'mM':
                final_arr = final_arr.view('int64')
            dtype = final_arr.dtype.kind
            func_name = 'get_group_assignment_' + utils.convert_kind_to_dtype(dtype) + '_1d'
            return getattr(_gb, func_name)(final_arr)
        elif len(self._group_dtype_loc) == 1 or 'O' not in self._group_dtype_loc:
            arrs = []
            for dtype, locs in self._group_dtype_loc.items():
                arr = self._df._data[dtype][:, locs]
                if dtype in 'mM':
                    arr = arr.view('int64')
                arrs.append(arr)
            if len(arrs) == 1:
                final_arr = arrs[0]
            else:
                final_arr = np.column_stack(arrs)

            dtype = final_arr.dtype.kind
            func_name = 'get_group_assignment_' + utils.convert_kind_to_dtype(dtype) + '_2d'
            final_arr = np.ascontiguousarray(final_arr)
            return getattr(_gb, func_name)(final_arr)
        else:
            arrs = []
            for dtype, locs in self._group_dtype_loc.items():
                if dtype == 'O':
                    arr_str = self._df._data['O'][:, locs]
                else:
                    arr = self._df._data[dtype][:, locs]
                    if dtype in 'mM':
                        arr = arr.view('int64')
                    arrs.append(arr)
            if len(arrs) == 1:
                arr_numbers = arrs[0]
            else:
                arr_numbers = np.column_stack(arrs)

            dtype = arr_numbers.dtype.kind
            if arr_str.shape[1] == 1:
                arr_str = arr_str[:, 0]
            if arr_numbers.shape[1] == 1:
                arr_numbers = arr_numbers[:, 0]

            str_ndim = str(arr_str.ndim) + 'd_'
            num_ndim = str(arr_numbers.ndim) + 'd'
            dtype_str = utils.convert_kind_to_dtype(dtype) + '_'
            func_name = 'get_group_assignment_str_' + str_ndim + dtype_str + num_ndim
            arr_numbers = np.ascontiguousarray(arr_numbers)
            return getattr(_gb, func_name)(arr_str, arr_numbers)

    def _get_group_col_data(self) -> Dict[str, List[ndarray]]:
        data_dict: Dict[str, List[ndarray]] = defaultdict(list)
        for dtype, locs in self._group_dtype_loc.items():
            ix = np.ix_(self._group_position, locs)
            arr = self._df._data[dtype][ix]
            if arr.ndim == 1:
                arr = arr[:, np.newaxis]
            data_dict[dtype].append(arr)
        return data_dict

    def _get_group_col_data_all(self) -> Dict[str, List[ndarray]]:
        data_dict: Dict[str, List[ndarray]] = defaultdict(list)
        for dtype, locs in self._group_dtype_loc.items():
            arr = self._df._data[dtype][:, locs]
            if arr.ndim == 1:
                arr = arr[:, np.newaxis]
            data_dict[dtype].append(arr)
        return data_dict

    def _get_agg_name(self, name: str) -> str:
        i = 1
        while name in self._group_columns:
            name = name + str(i)
        return name

    def __repr__(self) -> str:
        return ("This is a groupby object. Here is some info on it:\n"
                f"Grouping Columns: {self._group_columns}\n"
                f"Number of Groups: {len(self._group_position)}")

    def __len__(self) -> int:
        return len(self._group_position)

    def _get_new_column_info(self) -> ColInfoT:
        new_column_info: ColInfoT = {}
        for col, col_obj in self._column_info.items():
            new_column_info[col] = utils.Column(*col_obj.values)
        return new_column_info

    @property
    def ngroups(self) -> int:
        return len(self._group_position)

    def _group_agg(self, name: str, ignore_str: bool = True, add_positions: bool = False,
                   keep_group_cols: bool = True, ignore_date: bool = True,
                   keep_date_type: bool = True, **kwargs) -> DataFrame:
        labels = self._group_labels
        size = len(self._group_position)

        old_dtype_col: Dict[str, List[str]] = defaultdict(list)
        for col, col_obj in self._df._column_info.items():
            if col not in self._group_columns:
                old_dtype_col[col_obj.dtype].append(col)

        if keep_group_cols:
            data_dict = self._get_group_col_data()
            new_column_info = self._get_new_column_info()
            new_columns = self._group_columns.copy()
        else:
            data_dict = defaultdict(list)
            new_column_info = {}
            new_columns = []

        for dtype, data in self._df._data.items():
            if ignore_str and dtype == 'O':
                continue
            if ignore_date and dtype in 'mM':
                continue
            # number of grouped columns
            group_locs: list = self._group_dtype_loc.get(dtype, [])
            if len(group_locs) != data.shape[1]:
                func_name = name + '_' + utils.convert_kind_to_dtype_generic(dtype)
                func = getattr(_gb, func_name)
                if dtype in 'mM':
                    data = data.view('int64')

                if add_positions:
                    arr = func(labels, size, data, group_locs, self._group_position, **kwargs)
                else:
                    arr = func(labels, size, data, group_locs, **kwargs)
            else:
                continue

            if dtype in 'mM' and keep_date_type:
                new_kind = dtype
                arr = arr.astype(utils.convert_kind_to_dtype(dtype))
            else:
                new_kind = arr.dtype.kind
            cur_loc = utils.get_num_cols(data_dict.get(new_kind, []))
            data_dict[new_kind].append(arr)

            for col in old_dtype_col[dtype]:
                count_less = 0
                old_kind, old_loc, old_order = self._df._column_info[col].values
                for k in self._group_dtype_loc.get(dtype, []):
                    count_less += old_loc > k

                new_column_info[col] = utils.Column(new_kind, cur_loc + old_loc - count_less, 0)

        i = len(new_columns)
        j = 0
        for col in self._df._columns:
            if col not in new_column_info:
                continue
            if col in self._group_columns and keep_group_cols:
                new_column_info[col].order = j
                j += 1
                continue

            new_columns.append(col)
            new_column_info[col].order = i
            i += 1

        new_data = utils.concat_stat_arrays(data_dict)

        return DataFrame._construct_from_new(new_data, new_column_info, new_columns)

    def size(self):
        name = self._get_agg_name('size')
        new_columns = np.array(self._group_columns + [name], dtype='O')
        size = _gb.size(self._group_labels, len(self._group_position))[:, np.newaxis]
        data_dict = self._get_group_col_data()
        data_dict['i'].append(size)
        new_data = utils.concat_stat_arrays(data_dict)
        new_column_info = self._get_new_column_info()
        new_column_info[name] = utils.Column('i', new_data['i'].shape[1] - 1,
                                             len(new_columns) - 1)
        return DataFrame._construct_from_new(new_data, new_column_info, new_columns)

    def count(self) -> DataFrame:
        return self._group_agg('count', ignore_str=False, ignore_date=False, keep_date_type=False)

    def cumcount(self) -> DataFrame:
        # todo: add ascending=False
        name = self._get_agg_name('cumcount')
        new_columns = np.array(self._group_columns + [name], dtype='O')
        cumcount = _gb.cumcount(self._group_labels, len(self._group_position))[:, np.newaxis]
        data_dict = self._get_group_col_data_all()
        data_dict['i'].append(cumcount)
        new_data = utils.concat_stat_arrays(data_dict)
        new_column_info = self._get_new_column_info()
        new_column_info[name] = utils.Column('i', new_data['i'].shape[1] - 1,
                                             len(new_columns) - 1)
        return DataFrame._construct_from_new(new_data, new_column_info, new_columns)

    def sum(self) -> DataFrame:
        return self._group_agg('sum')

    def prod(self) -> DataFrame:
        return self._group_agg('prod')

    def mean(self) -> DataFrame:
        return self._group_agg('mean')

    def max(self) -> DataFrame:
        return self._group_agg('max', False, ignore_date=False)

    def min(self) -> DataFrame:
        return self._group_agg('min', False, ignore_date=False)

    def first(self) -> DataFrame:
        new_columns = self._group_columns.copy()
        for col in self._df._columns:
            if col in self._group_columns:
                continue
            new_columns.append(col)
        return self._df[self._group_position, new_columns]

    def last(self) -> DataFrame:
        return self._group_agg('last', False)

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

    def median(self) -> DataFrame:
        return self._group_agg('median')

    def nunique(self) -> DataFrame:
        return self._group_agg('nunique', False, ignore_date=False, keep_date_type=False)

    def head(self, n=5) -> DataFrame:
        row_idx = _gb.head(self._group_labels, len(self), n=n)
        return self._df[row_idx, :]

    def tail(self, n=5) -> DataFrame:
        row_idx = _gb.tail(self._group_labels, len(self), n=n)
        return self._df[row_idx, :]

    def cummax(self) -> DataFrame:
        return self._group_agg('cummax', keep_group_cols=False, ignore_date=False)

    def cummin(self) -> DataFrame:
        return self._group_agg('cummin', keep_group_cols=False, ignore_date=False)

    def cumsum(self) -> DataFrame:
        return self._group_agg('cumsum', keep_group_cols=False)

    def cumprod(self) -> DataFrame:
        return self._group_agg('cumprod', keep_group_cols=False)

    def _single_agg(self, agg_cols: Dict = None, new_names: Dict = None,
                    new_order: Dict = None, num_agg_cols: int = None,
                    func_kwargs: Dict = None) -> DataFrame:

        labels = self._group_labels
        size = len(self._group_position)

        data_dict = self._get_group_col_data()
        new_column_info = self._get_new_column_info()
        new_columns = self._group_columns.copy() + [''] * num_agg_cols

        for name, agg_cols in agg_cols.items():

            agg_dtype_locs = defaultdict(list)
            agg_dtype_names = defaultdict(list)
            agg_dtype_new_names = defaultdict(list)
            agg_dtype_order = defaultdict(list)
            non_agg_dtype_locs = defaultdict(list)
            agg_dtype_kwargs = defaultdict(list)

            if isinstance(name, str):
                # name can also be a custom function
                name_kwargs = get_func_kwargs(name)
                ignore_str = name_kwargs.get('ignore_str', True)
                add_positions = name_kwargs.get('add_positions', False)
                ignore_date = name_kwargs.get('ignore_date', True)
                keep_date_type = name_kwargs.get('keep_date_type', True)
            else:
                ignore_str = False
                add_positions = False
                ignore_date = False
                keep_date_type = True

            cur_new_names = new_names[name]
            cur_new_order = new_order[name]
            kwargs_list = func_kwargs[name]

            for col in self._df._columns:

                dtype, loc, _ = self._df._column_info[col].values
                try:
                    idx = agg_cols.index(col)
                except ValueError:
                    non_agg_dtype_locs[dtype].append(loc)
                else:
                    agg_dtype_locs[dtype].append(loc)
                    agg_dtype_names[dtype].append(col)
                    agg_dtype_new_names[dtype].append(cur_new_names[idx])
                    agg_dtype_order[dtype].append(cur_new_order[idx])
                    agg_dtype_kwargs[dtype].append(kwargs_list[idx])

            for dtype, data in self._df._data.items():
                if dtype not in agg_dtype_locs:
                    continue
                if ignore_str and dtype == 'O':
                    continue
                if ignore_date and dtype in 'mM':
                    continue

                if dtype in 'mM':
                    data = data.view('int64')

                kwargs = {}
                for kw in agg_dtype_kwargs[dtype]:
                    if kw is not None:
                        kwargs = kw
                        break

                if isinstance(name, str):
                    func_name = name + '_' + utils.convert_kind_to_dtype_generic(dtype)
                else:
                    func_name = 'custom_' + utils.convert_kind_to_dtype_generic(dtype)
                    # 'name' is actually a function here
                    kwargs['func'] = name
                    kwargs['col_dict'] = dict(zip(agg_dtype_locs[dtype], agg_dtype_names[dtype]))

                func = getattr(_gb, func_name)

                if add_positions:
                    arr = func(labels, size, data, non_agg_dtype_locs[dtype], self._group_position, **kwargs)
                else:
                    arr = func(labels, size, data, non_agg_dtype_locs[dtype], **kwargs)

                if dtype in 'mM' and keep_date_type:
                    new_kind = dtype
                    arr = arr.astype(utils.convert_kind_to_dtype(dtype))
                else:
                    new_kind = arr.dtype.kind

                cur_loc = utils.get_num_cols(data_dict.get(new_kind, []))
                data_dict[new_kind].append(arr)

                old_locs = agg_dtype_locs[dtype]
                order = np.argsort(old_locs).tolist()

                cur_names = np.array(agg_dtype_new_names[dtype])[order]
                cur_order = len(self._group_columns) + np.array(agg_dtype_order[dtype])[order]

                for i, cur_name in enumerate(cur_names):
                    new_column_info[cur_name] = utils.Column(new_kind, cur_loc + i, cur_order[i])
                    new_columns[cur_order[i]] = cur_name

        new_data = utils.concat_stat_arrays(data_dict)
        new_columns = np.array(new_columns, dtype='O')
        return DataFrame._construct_from_new(new_data, new_column_info, new_columns)

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
                # TODO - the second argument below needs to be gotten from aggregating column.
                # Its simply defaulted now
                utils.validate_agg_func(arg[0], 'i')
            elif not isinstance(arg[0], Callable):
                raise TypeError('The first item of the tuple must be an aggregating function name '
                                'as a string or a user-defined function')

            if not isinstance(arg[1], str):
                raise TypeError('The second element in each tuple must be a column name as a '
                                'string.')
            elif arg[1] not in self._df._column_info:
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
        result = _gb.filter(labels, size, self._df, func, *args, **kwargs)

        new_data = {kind: data[result] for kind, data in self._df._data.items()}
        new_col_info = self._df._copy_column_info()
        columns = self._df._columns.copy()

        return self._df._construct_from_new(new_data, new_col_info, columns)

    def apply(self, func, *args, **kwargs):
        if not isinstance(func, Callable):
            raise TypeError('The `func` variable must be a function or any callable object')
        labels = self._group_labels
        size = len(self._group_position)
        new_data, new_column_info, new_columns, group_repeats = _gb.apply(labels, size, self._df, func, *args, **kwargs)

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
