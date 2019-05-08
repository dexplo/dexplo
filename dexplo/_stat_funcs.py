from typing import Any

import numpy as np
from numpy import ndarray

from ._libs import math as m
from ._libs import math_oper_string as mos
from . import _math_columns as mc
from . import _date_funcs as df


def _nanpercentile(a: ndarray, q: float, axis: int, **kwargs: Any) -> ndarray:
    return np.nanpercentile(a, q, axis)


funcs = {'i': {'min': m.min_int,
               'max': m.max_int,
               'sum': m.sum_int,
               'mean': m.mean_int,
               'median': m.median_int,
               'mode': m.mode_int,
               'prod': m.prod_int,
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
               'cumprod': m.cumprod_int,
               'quantile': m.quantile_int,
               'nunique': m.nunique_int},

         'b': {'min': m.min_bool,
               'max': m.max_bool,
               'sum': m.sum_bool,
               'mean': m.mean_bool,
               'median': m.median_bool,
               'mode': m.mode_bool,
               'prod': m.prod_bool,
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
               'cumprod': m.cumprod_bool,
               'quantile': m.quantile_bool,
               'nunique': m.nunique_bool},

         'f': {'min': m.min_float,
               'max': m.max_float,
               'sum': m.sum_float,
               'mean': m.mean_float,
               'median': m.median_float,
               'mode': m.mode_float,
               'prod': m.prod_float,
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
               'cumprod': m.cumprod_float,
               'quantile': m.quantile_float,
               'nunique': m.nunique_float},

         'S': {'min': m.min_str,
               'max': m.max_str,
               'mode': m.mode_str,
               'sum': m.sum_str,
               'any': m.any_str,
               'all': m.all_str,
               'argmax': m.argmax_str,
               'argmin': m.argmin_str,
               'count': m.count_str,
               'cummin': m.cummin_str,
               'cummax': m.cummax_str,
               'cumsum': m.cumsum_str,
               'nunique': m.nunique_str},

         'M': {'min': df.min_date,
               'max': df.max_date,
               'any': df.any_date,
               'all': df.all_date,
               'argmax': df.argmax_date,
               'argmin': df.argmin_date,
               'count': df.count_date,
               'cummax': df.cummax_date,
               'cummin': df.cummin_date,
               'mode': df.mode_date,
               'nunique': df.nunique_date},

         'm': {'min': df.min_date,
               'max': df.max_date,
               'sum': df.sum_date,
               'mean': df.mean_date,
               'median': df.median_date,
               'prod': df.prod_date,
               'any': df.any_date,
               'all': df.all_date,
               'argmax': df.argmax_date,
               'argmin': df.argmin_date,
               'count': df.count_date,
               'cummax': df.cummax_date,
               'cummin': df.cummin_date,
               'cumsum': df.cumsum_date,
               'cumprod': df.cumprod_date,
               'mode': df.mode_date,
               'nunique': df.nunique_date}}

funcs_columns = {'sum': mc.sum_columns,
                 'max': mc.max_columns,
                 'min': mc.min_columns,
                 'mean': mc.mean_columns,
                 'prod': mc.prod_columns,
                 'any': mc.any_columns,
                 'all': mc.all_columns,
                 'count': mc.count_columns}

funcs_str = {'__add__': mos.add_str,
             '__radd__': mos.radd_str,
             '__lt__': mos.lt_str,
             '__le__': mos.le_str,
             '__gt__': mos.gt_str,
             '__ge__': mos.ge_str,
             '__eq__': mos.eq_str,
             '__ne__': mos.ne_str,
             '__mul__': mos.mul_str,
             '__rmul__': mos.mul_str}

funcs_str1 = {'__add__': mos.add_str_one,
              '__lt__': mos.lt_str_one,
              '__le__': mos.le_str_one,
              '__gt__': mos.gt_str_one,
              '__ge__': mos.ge_str_one,
              '__eq__': mos.eq_str_one,
              '__ne__': mos.ne_str_one,
              '__mul__': mos.mul_str_one,
              '__rmul__': mos.mul_str_one}

funcs_str2 = {'__add__': mos.add_str_two,
              '__lt__': mos.lt_str_two,
              '__le__': mos.le_str_two,
              '__gt__': mos.gt_str_two,
              '__ge__': mos.ge_str_two,
              '__eq__': mos.eq_str_two,
              '__ne__': mos.ne_str_two,
              '__mul__': mos.mul_str_two,
              '__rmul__': mos.mul_str_two}

funcs_str2_1row_right = {'__add__': mos.add_str_two_1row_right,
                         '__radd__': mos.add_str_two_1row_left,
                         '__lt__': mos.lt_str_two_1row_right,
                         '__le__': mos.le_str_two_1row_right,
                         '__gt__': mos.gt_str_two_1row_right,
                         '__ge__': mos.ge_str_two_1row_right,
                         '__eq__': mos.eq_str_two_1row_right,
                         '__ne__': mos.ne_str_two_1row_right,}

funcs_str2_1row_left = {'__add__': mos.add_str_two_1row_left,
                        '__radd__': mos.add_str_two_1row_right,
                        '__lt__': mos.lt_str_two_1row_left,
                        '__le__': mos.le_str_two_1row_left,
                        '__gt__': mos.gt_str_two_1row_left,
                        '__ge__': mos.ge_str_two_1row_left,
                        '__eq__': mos.eq_str_two_1row_left,
                        '__ne__': mos.ne_str_two_1row_left,}

def still_string(name):
    return name in {'__add__', '__radd__', '__mul__', '__rmul__'}
