from typing import List
import numpy as np
from numpy import ndarray


def sum_columns(arrs: List[ndarray]) -> ndarray:
    arr: ndarray = arrs[0] + arrs[1]
    arr_other: ndarray
    for arr_other in arrs[2:]:
        arr += arr_other
    return arr


def max_columns(arrs: List[ndarray]) -> ndarray:
    return np.nanmax(arrs, 0)


def min_columns(arrs: List[ndarray]) -> ndarray:
    return np.nanmin(arrs, 0)


def mean_columns(arrs: List[ndarray]) -> ndarray:
    return np.nanmean(arrs, 0)


def any_columns(arrs: List[ndarray]) -> ndarray:
    return np.any(arrs, 0)


def all_columns(arrs: List[ndarray]) -> ndarray:
    return np.all(arrs, 0)


def count_columns(arrs: List[ndarray]) -> ndarray:
    return np.sum(arrs, 0)
