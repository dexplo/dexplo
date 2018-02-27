import numpy as np
import dexplo._libs.math as _math


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