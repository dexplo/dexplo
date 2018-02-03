import dexplo._libs.math as m
import dexplo.math_columns as mc
import numpy as np
from numpy import ndarray
from typing import Any


def _nanpercentile(a: ndarray, q: float, axis: int, **kwargs: Any) -> ndarray:
    return np.nanpercentile(a, q, axis)


funcs = {'i': {'min': m.min_int,
               'max': m.max_int,
               'sum': m.sum_int,
               'mean': m.mean_int,
               'median': m.median_int,
               'std': m.std_int,
               'var': m.var_int,
               'any': m.any_int,
               'all': m.all_int,
               'argmax': m.argmax_int,
               'argmin': m.argmin_int,
               'count': m.count_int,
               'cummin': m.cummin_int,
               'cummax': m.cummax_int,
               'cumsum': m.cumsum_int,
               'quantile': m.quantile_int,
               'nunique': m.nunique_int},

         'b': {'min': m.min_bool,
               'max': m.max_bool,
               'sum': m.sum_bool,
               'mean': m.mean_bool,
               'median': m.median_bool,
               'std': m.std_bool,
               'var': m.var_bool,
               'any': m.any_bool,
               'all': m.all_bool,
               'argmax': m.argmax_bool,
               'argmin': m.argmin_bool,
               'count': m.count_bool,
               'cummin': m.cummin_bool,
               'cummax': m.cummax_bool,
               'cumsum': m.cumsum_bool,
               'quantile': m.quantile_bool,
               'nunique': m.nunique_bool},

         'f': {'min': m.min_float,
               'max': m.max_float,
               'sum': m.sum_float,
               'mean': m.mean_float,
               'median': m.median_float,
               'std': m.std_float,
               'var': m.var_float,
               'any': m.any_float,
               'all': m.all_float,
               'argmax': m.argmax_float,
               'argmin': m.argmin_float,
               'count': m.count_float,
               'cummin': m.cummin_float,
               'cummax': m.cummax_float,
               'cumsum': m.cumsum_float,
               'quantile': m.quantile_float,
               'nunique': m.nunique_float},

         'O': {'min': m.min_str,
               'max': m.max_str,
               'sum': m.sum_str,
               'any': m.any_str,
               'all': m.all_str,
               'argmax': m.argmax_str,
               'argmin': m.argmin_str,
               'count': m.count_str,
               'cummin': m.cummin_str,
               'cummax': m.cummax_str,
               'cumsum': m.cumsum_str,
               'nunique': m.nunique_str}}

funcs_columns = {'sum': mc.sum_columns,
                 'max': mc.max_columns,
                 'min': mc.min_columns,
                 'mean': mc.mean_columns,
                 'any': mc.any_columns,
                 'all': mc.all_columns,
                 'count': mc.count_columns}

funcs_str = {'__add__': m.add_obj,
             '__radd__': m.radd_obj,
             '__lt__': m.lt_obj,
             '__le__': m.le_obj,
             '__gt__': m.gt_obj,
             '__ge__': m.ge_obj,
             '__mul__': m.mul_obj,
             '__rmul__': m.mul_obj}
