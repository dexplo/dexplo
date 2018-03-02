from dexplo._frame import DataFrame
import dexplo._utils as utils
from collections import defaultdict
import numpy as np
from numpy import ndarray
from typing import Union, Dict, List, Tuple
from dexplo._libs import groupby as _gb

ColInfoT = Dict[str, utils.Column]


class Grouper(object):

    def __init__(self, df: DataFrame, columns: List[str]) -> None:
        self._df: DataFrame = df
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
                   keep_group_cols: bool = True, ignore_date: bool = True, keep_date_type=True,
                   **kwargs) -> DataFrame:
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