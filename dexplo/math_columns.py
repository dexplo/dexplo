import numpy as np


def sum_columns(arrs):
    arr = arrs[0] + arrs[1]
    for arr_other in arrs[2:]:
        arr += arr_other
    return arr


def max_columns(arrs):
    return np.max(arrs, 0)


def min_columns(arrs):
    return np.min(arrs, 0)


def mean_columns(arrs):
    return np.mean(arrs, 0)


def any_columns(arrs):
    return np.any(arrs, 0)


def all_columns(arrs):
    return np.all(arrs, 0)


def count_columns(arrs):
    return np.sum(arrs, 0)
