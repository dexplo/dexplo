import numpy as np
from ._libs import math as _math
from . import _utils

def max_date(arr, axis, **kwargs):
    return arr.max(axis=axis)

def min_date(arr, axis, **kwargs):
    return arr.min(axis=axis)

def any_date(arr, axis, **kwargs):
    return (~np.isnat(arr)).sum(axis=axis) > 0

def all_date(arr, axis, **kwargs):
    return (~np.isnat(arr)).sum(axis=axis) == arr.shape[0]

def argmax_date(arr, axis, **kwargs):
    return arr.argmax(axis=axis)

def argmin_date(arr, axis, **kwargs):
    return arr.argmin(axis=axis)

def count_date(arr, axis, **kwargs):
    return (~np.isnat(arr)).sum(axis=axis)

def cummax_date(arr, axis, **kwargs):
    return np.maximum.accumulate(arr, axis=axis)

def cummin_date(arr, axis, **kwargs):
    return np.minimum.accumulate(arr, axis=axis)

def nunique_date(arr, axis, **kwargs):
    return _math.nunique_int(arr.view('int64'), axis=axis)

def mode_date(arr, axis, **kwargs):
    kind = arr.dtype.kind
    return _math.mode_int(arr.view('int64'), axis=axis, **kwargs).astype(_utils._DT[kind])


## These below will only work for timedeltas

def sum_date(arr, axis, **kwargs):
    return arr.sum(axis=axis)

def median_date(arr, axis, **kwargs):
    return np.median(arr, axis=axis)

def mean_date(arr, axis, **kwargs):
    return arr.mean(axis=axis)

def prod_date(arr, axis, **kwargs):
    return arr.prod(axis=axis)

def cumsum_date(arr, axis, **kwargs):
    return np.cumsum(arr, axis=axis)

def cumprod_date(arr, axis, **kwargs):
    return np.cumprod(arr, axis=axis)
