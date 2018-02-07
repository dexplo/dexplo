#cython: boundscheck=False
#cython: wraparound=False
import numpy as np
cimport numpy as np
from numpy cimport ndarray
from collections import defaultdict
import cython
from cpython cimport dict, set, list, tuple
from libc.math cimport isnan, sqrt
from numpy import nan
from .math import min_max_int, min_max_int2, isna_str, get_first_non_nan
from libc.stdlib cimport malloc, free
try:
    import bottleneck as bn
except ImportError:
    import numpy as bn

MAX_FLOAT = np.finfo(np.float64).max
MIN_FLOAT = np.finfo(np.float64).min

MAX_INT = np.iinfo(np.int64).max
MIN_INT = np.iinfo(np.int64).min

MAX_CHAR = chr(1_000_000)
MIN_CHAR = chr(0)


def get_group_assignment_str_1d(ndarray[object] a):
    cdef int i, j, k
    cdef int nr = a.shape[0]
    cdef int nc = a.shape[1]
    cdef int count = 0
    cdef ndarray[np.int64_t] group = np.empty(nr, dtype=np.int64)
    cdef ndarray[np.int64_t] group_position = np.empty(nr, dtype=np.int64)
    cdef ndarray[object] qq = np.empty(nc, dtype='O')
    cdef dict d = {}
    cdef tuple t

    for i in range(nr):
        group[i] = d.get(a[i], -1)
        if group[i] == -1:
            group_position[count] = i
            group[i] = count
            d[a[i]] = count
            count += 1

    return group, group_position[:count]

def get_group_assignment_str_2d(ndarray[object, ndim=2] a):
    cdef int i, j, k
    cdef int nr = a.shape[0]
    cdef int nc = a.shape[1]
    cdef int count = 0
    cdef ndarray[np.int64_t] group = np.empty(nr, dtype=np.int64)
    cdef ndarray[np.int64_t] group_position = np.empty(nr, dtype=np.int64)
    cdef ndarray[object] qq = np.empty(nc, dtype='O')
    cdef dict d = {}
    cdef tuple t

    for i in range(nr):
        if nc == 2:
            t = (a[i, 0], a[i, 1])
        elif nc == 3:
            t = (a[i, 0], a[i, 1], a[i, 2])
        elif nc == 4:
            t = (a[i, 0], a[i, 1], a[i, 2], a[i, 3])
        elif nc == 5:
            t = (a[i, 0], a[i, 1], a[i, 2], a[i, 3], a[i, 4])
        elif nc == 6:
            t = (a[i, 0], a[i, 1], a[i, 2], a[i, 3], a[i, 4], a[i, 5])
        elif nc == 7:
            t = (a[i, 0], a[i, 1], a[i, 2], a[i, 3], a[i, 4], a[i, 5], a[i, 6])
        else:
            t = tuple(a)

        group[i] = d.get(t, -1)
        if group[i] == -1:
            group_position[count] = i
            group[i] = count
            d[t] = count
            count += 1

    return group, group_position[:count]

def get_group_assignment_int_1d(ndarray[np.int64_t] a):
    cdef int i
    cdef int n = len(a)
    cdef int count = 0
    cdef ndarray[np.int64_t] group = np.empty(n, dtype=np.int64)
    cdef ndarray[np.int64_t] group_position = np.empty(n, dtype=np.int64)
    cdef dict d = {}

    low, high = min_max_int(a)
    if high - low < 10_000_000:
        return get_group_assignment_int_bounded(a, low, high)

    for i in range(n):
        group[i] = d.get(a[i], -1)
        if group[i] == -1:
            group_position[count] = i
            group[i] = count
            d[a[i]] = count
            count += 1
    return group, group_position[:count]

def get_group_assignment_int_bounded(ndarray[np.int64_t] a, np.int64_t low, np.int64_t high):
    cdef int i
    cdef count = 0
    cdef int n = len(a)
    cdef ndarray[np.int64_t] unique
    cdef np.int64_t rng

    cdef ndarray[np.int64_t] group = np.empty(n, dtype=np.int64)
    cdef ndarray[np.int64_t] group_position = np.empty(n, dtype=np.int64)

    rng = high - low + 1
    unique = np.full(rng, -1, dtype='int64')

    for i in range(n):
        if unique[a[i] - low] == -1:
            # first time a group appears
            unique[a[i] - low] = count
            group_position[count] = i
            group[i] = count
            count += 1
        else:
            group[i] = unique[a[i] - low]
    return group, group_position[:count]

def get_group_assignment_int_2d(ndarray[np.int64_t, ndim=2] a):
    cdef int i
    cdef int n = len(a)
    cdef int nc = a.shape[1]
    cdef int count = 0
    cdef ndarray[np.int64_t] group = np.empty(n, dtype=np.int64)
    cdef ndarray[np.int64_t] group_position = np.empty(n, dtype=np.int64)
    cdef dict d = {}
    cdef tuple t
    cdef ndarray[np.int64_t] ranges
    cdef np.int64_t total_range

    lows, highs = min_max_int2(a, 0)

    ranges = highs - lows + 1
    total_range = np.prod(ranges)

    if total_range < 10_000_000:
        return get_group_assignment_int_bounded_2d(a, lows, highs, ranges, total_range)

    if nc == 2:
        for i in range(n):
            t = (a[i, 0], a[i, 1])
            group[i] = d.get(t, -1)
            if group[i] == -1:
                group_position[count] = i
                group[i] = count
                d[t] = count
                count += 1

    elif nc == 3:
        for i in range(n):
            t = (a[i, 0], a[i, 1], a[i, 2])
            group[i] = d.get(t, -1)
            if group[i] == -1:
                group_position[count] = i
                group[i] = count
                d[t] = count
                count += 1

    elif nc == 4:
        for i in range(n):
            t = (a[i, 0], a[i, 1], a[i, 2], a[i, 3])
            group[i] = d.get(t, -1)
            if group[i] == -1:
                group_position[count] = i
                group[i] = count
                d[t] = count
                count += 1

    elif nc == 5:
        for i in range(n):
            t = (a[i, 0], a[i, 1], a[i, 2], a[i, 3], a[i, 4])
            group[i] = d.get(t, -1)
            if group[i] == -1:
                group_position[count] = i
                group[i] = count
                d[t] = count
                count += 1
    elif nc == 6:
        for i in range(n):
            t = (a[i, 0], a[i, 1], a[i, 2], a[i, 3], a[i, 4], a[i, 5])
            group[i] = d.get(t, -1)
            if group[i] == -1:
                group_position[count] = i
                group[i] = count
                d[t] = count
                count += 1
    else:
        for i in range(n):
            t = tuple(a)
            group[i] = d.get(t, -1)
            if group[i] == -1:
                group_position[count] = i
                group[i] = count
                d[t] = count
                count += 1

    return group, group_position[:count]

def get_group_assignment_int_bounded_2d(ndarray[np.int64_t, ndim=2] a, ndarray[np.int64_t] lows,
                                        ndarray[np.int64_t] highs, ndarray[np.int64_t] ranges, int total_range):
    cdef int i, j
    cdef count = 0
    cdef int n = len(a)
    cdef int nc = a.shape[1]

    cdef ndarray[np.int64_t] group = np.empty(n, dtype=np.int64)
    cdef ndarray[np.int64_t] group_position = np.empty(n, dtype=np.int64)
    cdef ndarray[np.int64_t, ndim=2] idx
    cdef long *unique = <long *>malloc(total_range * sizeof(long))
    cdef long iloc
    cdef ndarray[np.int64_t] range_prod


    for i in range(total_range):
        unique[i] = -1

    idx = a - lows
    range_prod = np.cumprod(ranges[:nc - 1])

    for i in range(n):
        iloc = idx[i, 0]
        for j in range(nc - 1):
            iloc += range_prod[j] * idx[i, j + 1]
        if unique[iloc] == -1:
            # first time a group appears
            unique[iloc] = count
            group_position[count] = i
            group[i] = count
            count += 1
        else:
            group[i] = unique[iloc]

    free(unique)

    return group, group_position[:count]


def get_group_assignment_float_1d(ndarray[np.float64_t] a):
    cdef int i
    cdef int n = len(a)
    cdef int count = 0
    cdef ndarray[np.int64_t] group = np.empty(n, dtype=np.int64)
    cdef ndarray[np.int64_t] group_position = np.empty(n, dtype=np.int64)
    cdef dict d = {}

    for i in range(n):
        if isnan(a[i]):
            v = None
        else:
            v = a[i]
        group[i] = d.get(v, -1)
        if group[i] == -1:
            group_position[count] = i
            group[i] = count
            d[v] = count
            count += 1
    return group, group_position[:count]


def get_group_assignment_float_2d(ndarray[np.float64_t, ndim=2] a):
    cdef int i, j
    cdef int n = len(a)
    cdef int nc = a.shape[1]
    cdef int count = 0
    cdef ndarray[np.int64_t] group = np.empty(n, dtype=np.int64)
    cdef ndarray[np.int64_t] group_position = np.empty(n, dtype=np.int64)
    cdef dict d = {}
    cdef tuple t

    if nc == 2:
        for i in range(n):
            if isnan(a[i, 0]):
                v0 = None
            else:
                v0 = a[i, 0]
            if isnan(a[i, 1]):
                v1 = None
            else:
                v1 = a[i, 1]
            t = (v0, v1)
            group[i] = d.get(t, -1)
            if group[i] == -1:
                group_position[count] = i
                group[i] = count
                d[t] = count
                count += 1
    elif nc == 3:
        for i in range(n):
            if isnan(a[i, 0]):
                v0 = None
            else:
                v0 = a[i, 0]
            if isnan(a[i, 1]):
                v1 = None
            else:
                v1 = a[i, 1]
            if isnan(a[i, 2]):
                v2 = None
            else:
                v2 = a[i, 2]

            t = (v0, v1, v2)
            group[i] = d.get(t, -1)
            if group[i] == -1:
                group_position[count] = i
                group[i] = count
                d[t] = count
                count += 1
    else:
        for i in range(n):
            v = []
            for j in range(nc):
                if isnan(a[i, j]):
                    v.append(None)
                else:
                    v.append(a[i, j])
            t = tuple(v)
            group[i] = d.get(t, -1)
            if group[i] == -1:
                group_position[count] = i
                group[i] = count
                d[t] = count
                count += 1

    return group, group_position[:count]


def get_group_assignment_bool_1d(ndarray[np.uint8_t, cast=True] a):
    cdef int i
    cdef int n = len(a)
    cdef int count = 0
    cdef ndarray[np.int64_t] group = np.empty(n, dtype=np.int64)
    cdef ndarray[np.int64_t] group_position = np.empty(n, dtype=np.int64)
    cdef ndarray[np.int64_t] unique = -np.ones(2, dtype='int64')
    cdef dict d = {}

    for i in range(n):
        if unique[a[i]] == -1:
            # first time a group appears
            unique[a[i]] = count
            group_position[count] = i
            group[i] = count
            count += 1
        else:
            group[i] = unique[a[i]]

    return group, group_position[:count]


def get_group_assignment_bool_2d(ndarray[np.uint8_t, cast=True, ndim=2] a):
    cdef int i, j, iloc
    cdef int n = len(a)
    cdef int nc = a.shape[1]
    cdef int count = 0
    cdef ndarray[np.int64_t] group = np.empty(n, dtype=np.int64)
    cdef ndarray[np.int64_t] group_position = np.empty(n, dtype=np.int64)
    cdef ndarray[np.int64_t] unique = -np.ones(2 ** nc, dtype='int64')

    for i in range(n):
        iloc = 0
        for j in range(nc):
            iloc += 2 ** j * a[i, j]
        if unique[iloc] == -1:
            # first time a group appears
            unique[iloc] = count
            group_position[count] = i
            group[i] = count
            count += 1
        else:
            group[i] = unique[iloc]

    return group, group_position[:count]


def size(ndarray[np.int64_t] a, int group_size):
    cdef int i
    cdef int n = len(a)
    cdef ndarray[np.int64_t] result = np.zeros(group_size, dtype=np.int64)
    for i in range(n):
        result[a[i]] += 1
    return result

def cumcount(ndarray[np.int64_t] a, int group_size):
    cdef int i
    cdef int n = len(a)
    cdef ndarray[np.int64_t] counter = np.zeros(group_size, dtype=np.int64)
    cdef ndarray[np.int64_t] result = np.empty(n, dtype=np.int64)

    for i in range(n):
        result[i] = counter[a[i]]
        counter[a[i]] += 1
    return result

def count_int(ndarray[np.int64_t] labels, int size, ndarray[np.int64_t, ndim=2] data, list group_locs):
    cdef int i, j
    cdef int nr = data.shape[0]
    cdef int nc = data.shape[1]
    cdef ndarray[np.int64_t, ndim=2] result = np.zeros((size, nc - len(group_locs)), dtype='int64')
    cdef int k = 0
    for i in range(nc):
        if i in group_locs:
            k += 1
            continue
        for j in range(nr):
            result[labels[j], i - k] += 1
    return result

def count_bool(ndarray[np.int64_t] labels, int size, ndarray[np.uint8_t, ndim=2, cast=True] data, list group_locs):
    cdef int i, j
    cdef int nr = data.shape[0]
    cdef int nc = data.shape[1]
    cdef ndarray[np.int64_t, ndim=2] result = np.zeros((size, nc - len(group_locs)), dtype='int64')
    cdef int k = 0
    for i in range(nc):
        if i in group_locs:
            k += 1
            continue
        for j in range(nr):
            result[labels[j], i - k] += 1
    return result

def count_float(ndarray[np.int64_t] labels, int size, ndarray[np.float64_t, ndim=2] data, list group_locs):
    cdef int i, j, k = 0
    cdef int nr = data.shape[0]
    cdef int nc = data.shape[1]
    cdef ndarray[np.int64_t, ndim=2] result = np.zeros((size, nc - len(group_locs)), dtype='int64')
    for i in range(nc):
        if i in group_locs:
            k += 1
            continue
        for j in range(nr):
            if not isnan(data[j, i]):
                result[labels[j], i - k] += 1
    return result

def count_str(ndarray[np.int64_t] labels, int size, ndarray[object, ndim=2] data, list group_locs):
    cdef int i, j
    cdef int nr = data.shape[0]
    cdef int nc = data.shape[1]
    cdef ndarray[np.int64_t, ndim=2] result = np.zeros((size, nc - len(group_locs)), dtype='int64')
    cdef int k = 0
    for i in range(nc):
        if i in group_locs:
            k += 1
            continue
        for j in range(nr):
            if data[j, i] is not None:
                result[labels[j], i - k] += 1
    return result

def sum_int(ndarray[np.int64_t] labels, int size, ndarray[np.int64_t, ndim=2] data, list group_locs):
    cdef int i, j, k = 0
    cdef int nr = data.shape[0]
    cdef int nc = data.shape[1]
    cdef ndarray[np.int64_t, ndim=2] result = np.zeros((size, nc - len(group_locs)), dtype='int64')
    for i in range(nc):
        if i in group_locs:
            k += 1
            continue
        for j in range(nr):
            result[labels[j], i - k] = result[labels[j], i - k] + data[j, i]
    return result

def sum_float(ndarray[np.int64_t] labels, int size, ndarray[np.float64_t, ndim=2] data, list group_locs):
    cdef int i, j, k = 0
    cdef int nr = data.shape[0]
    cdef int nc = data.shape[1]
    cdef ndarray[np.float64_t, ndim=2] result = np.zeros((size, nc - len(group_locs)), dtype='float64')
    for i in range(nc):
        if i in group_locs:
            k += 1
            continue
        for j in range(nr):
            if not isnan(data[j, i]):
                result[labels[j], i - k] = result[labels[j], i - k] + data[j, i]
    return result

def sum_bool(ndarray[np.int64_t] labels, int size, ndarray[np.uint8_t, ndim=2, cast=True] data, list group_locs):
    cdef int i, j, k = 0
    cdef int nr = data.shape[0]
    cdef int nc = data.shape[1]
    cdef ndarray[np.int64_t, ndim=2] result = np.zeros((size, nc - len(group_locs)), dtype='int64')
    for i in range(nc):
        if i in group_locs:
            k += 1
            continue
        for j in range(nr):
            result[labels[j], i - k] = result[labels[j], i - k] + data[j, i]
    return result

def sum_str(ndarray[np.int64_t] labels, int size, ndarray[object, ndim=2] data, list group_locs):
    cdef int i, j, k = 0
    cdef int nr = data.shape[0]
    cdef int nc = data.shape[1]
    cdef ndarray[object, ndim=2] result = np.zeros((size, nc - len(group_locs)), dtype='U').astype('O')
    for i in range(nc):
        if i in group_locs:
            k += 1
            continue
        for j in range(nr):
            if data[j, i] is not None:
                result[labels[j], i - k] = result[labels[j], i - k] + data[j, i]
    return result

def mean_int(ndarray[np.int64_t] labels, int size, ndarray[np.int64_t, ndim=2] data, list group_locs):
    cdef int i, j, k = 0
    cdef int nr = data.shape[0]
    cdef int nc = data.shape[1]
    cdef ndarray[np.int64_t, ndim=2] result = np.zeros((size, nc - len(group_locs)), dtype='int64')
    cdef ndarray[np.int64_t, ndim=2] counts = np.zeros((size, nc - len(group_locs)), dtype='int64')
    for i in range(nc):
        if i in group_locs:
            k += 1
            continue
        for j in range(nr):
            result[labels[j], i - k] = result[labels[j], i - k] + data[j, i]
            counts[labels[j], i - k] += 1
    return result / counts

def mean_float(ndarray[np.int64_t] labels, int size, ndarray[np.float64_t, ndim=2] data, list group_locs):
    cdef int i, j, k = 0
    cdef int nr = data.shape[0]
    cdef int nc = data.shape[1]
    cdef ndarray[np.float64_t, ndim=2] result = np.zeros((size, nc - len(group_locs)), dtype='float64')
    cdef ndarray[np.int64_t, ndim=2] counts = np.zeros((size, nc - len(group_locs)), dtype='int64')
    for i in range(nc):
        if i in group_locs:
            k += 1
            continue
        for j in range(nr):
            if not isnan(data[j, i]):
                result[labels[j], i - k] = result[labels[j], i - k] + data[j, i]
                counts[labels[j], i - k] += 1
    return result / counts

def mean_bool(ndarray[np.int64_t] labels, int size, ndarray[np.uint8_t, ndim=2, cast=True] data, list group_locs):
    cdef int i, j, k = 0
    cdef int nr = data.shape[0]
    cdef int nc = data.shape[1]
    cdef ndarray[np.int64_t, ndim=2] result = np.zeros((size, nc - len(group_locs)), dtype='int64')
    cdef ndarray[np.int64_t, ndim=2] counts = np.zeros((size, nc - len(group_locs)), dtype='int64')
    for i in range(nc):
        if i in group_locs:
            k += 1
            continue
        for j in range(nr):
            result[labels[j], i - k] = result[labels[j], i - k] + data[j, i]
            counts[labels[j], i - k] += 1
    return result / counts

def max_int(ndarray[np.int64_t] labels, int size, ndarray[np.int64_t, ndim=2] data, list group_locs):
    cdef int i, j, k = 0
    cdef int nr = data.shape[0]
    cdef int nc = data.shape[1]
    cdef ndarray[np.int64_t, ndim=2] result = np.full((size, nc - len(group_locs)), MIN_INT, dtype='int64')
    for i in range(nc):
        if i in group_locs:
            k += 1
            continue
        for j in range(nr):
            if data[j, i] > result[labels[j], i - k]:
                result[labels[j], i - k] = data[j, i]
    return result

def max_float(ndarray[np.int64_t] labels, int size, ndarray[np.float64_t, ndim=2] data, list group_locs):
    cdef int i, j, k = 0
    cdef int nr = data.shape[0]
    cdef int nc = data.shape[1]
    cdef ndarray[np.float64_t, ndim=2] result = np.full((size, nc - len(group_locs)), nan, dtype='float64')
    for i in range(nc):
        if i in group_locs:
            k += 1
            continue
        for j in range(nr):
            if not isnan(data[j, i]) and (isnan(result[labels[j], i - k]) or data[j, i] > result[labels[j], i - k]):
                result[labels[j], i - k] = data[j, i]
    return result

def max_bool(ndarray[np.int64_t] labels, int size, ndarray[np.uint8_t, ndim=2, cast=True] data, list group_locs):
    cdef int i, j, k = 0
    cdef int nr = data.shape[0]
    cdef int nc = data.shape[1]
    cdef ndarray[np.uint8_t, ndim=2, cast=True] result = np.zeros((size, nc - len(group_locs)), dtype='bool')
    for i in range(nc):
        if i in group_locs:
            k += 1
            continue
        for j in range(nr):
            if data[j, i] > result[labels[j], i - k]:
                result[labels[j], i - k] = data[j, i]
    return result

def max_str(ndarray[np.int64_t] labels, int size, ndarray[object, ndim=2] data, list group_locs):
    cdef int i, j, k = 0
    cdef int nr = data.shape[0]
    cdef int nc = data.shape[1]
    cdef ndarray[object, ndim=2] result = np.full((size, nc - len(group_locs)), None, dtype='O')
    for i in range(nc):
        if i in group_locs:
            k += 1
            continue
        for j in range(nr):
            if data[j, i] is not None:
                if result[labels[j], i - k] is None or data[j, i] > result[labels[j], i - k]:
                    result[labels[j], i - k] = data[j, i]
    return result

def min_int(ndarray[np.int64_t] labels, int size, ndarray[np.int64_t, ndim=2] data, list group_locs):
    cdef int i, j, k = 0
    cdef int nr = data.shape[0]
    cdef int nc = data.shape[1]
    cdef ndarray[np.int64_t, ndim=2] result = np.full((size, nc - len(group_locs)), MAX_INT, dtype='int64')
    for i in range(nc):
        if i in group_locs:
            k += 1
            continue
        for j in range(nr):
            if data[j, i] < result[labels[j], i - k]:
                result[labels[j], i - k] = data[j, i]
    return result

def min_float(ndarray[np.int64_t] labels, int size, ndarray[np.float64_t, ndim=2] data, list group_locs):
    cdef int i, j, k = 0
    cdef int nr = data.shape[0]
    cdef int nc = data.shape[1]
    cdef ndarray[np.float64_t, ndim=2] result = np.full((size, nc - len(group_locs)), nan, dtype='float64')
    for i in range(nc):
        if i in group_locs:
            k += 1
            continue
        for j in range(nr):
            if not isnan(data[j, i]) and (isnan(result[labels[j], i - k]) or data[j, i] < result[labels[j], i - k]):
                result[labels[j], i - k] = data[j, i]
    return result

def min_bool(ndarray[np.int64_t] labels, int size, ndarray[np.uint8_t, ndim=2, cast=True] data, list group_locs):
    cdef int i, j, k = 0
    cdef int nr = data.shape[0]
    cdef int nc = data.shape[1]
    cdef ndarray[np.uint8_t, ndim=2, cast=True] result = np.zeros((size, nc - len(group_locs)), dtype='bool')
    for i in range(nc):
        if i in group_locs:
            k += 1
            continue
        for j in range(nr):
            if data[j, i] < result[labels[j], i - k]:
                result[labels[j], i - k] = data[j, i]
    return result

def min_str(ndarray[np.int64_t] labels, int size, ndarray[object, ndim=2] data, list group_locs):
    cdef int i, j, k = 0
    cdef int nr = data.shape[0]
    cdef int nc = data.shape[1]
    cdef ndarray[object, ndim=2] result = np.full((size, nc - len(group_locs)), None, dtype='O')
    for i in range(nc):
        if i in group_locs:
            k += 1
            continue
        for j in range(nr):
            if data[j, i] is not None:
                if result[labels[j], i - k] is None or data[j, i] < result[labels[j], i - k]:
                    result[labels[j], i - k] = data[j, i]
    return result


def last_int(ndarray[np.int64_t] labels, int size, ndarray[np.int64_t, ndim=2] data, list group_locs):
    cdef int i, j, k = 0
    cdef int nr = data.shape[0]
    cdef int nc = data.shape[1]
    cdef ndarray[np.int64_t, ndim=2] result = np.empty((size, nc - len(group_locs)), dtype='int64')
    for i in range(nc):
        if i in group_locs:
            k += 1
            continue
        for j in range(nr):
            result[labels[j], i - k] = data[j, i]
    return result

def last_float(ndarray[np.int64_t] labels, int size, ndarray[np.float64_t, ndim=2] data, list group_locs):
    cdef int i, j, k = 0
    cdef int nr = data.shape[0]
    cdef int nc = data.shape[1]
    cdef ndarray[np.float64_t, ndim=2] result = np.empty((size, nc - len(group_locs)), dtype='float64')
    for i in range(nc):
        if i in group_locs:
            k += 1
            continue
        for j in range(nr):
            result[labels[j], i - k] = data[j, i]
    return result

def last_bool(ndarray[np.int64_t] labels, int size, ndarray[np.uint8_t, ndim=2, cast=True] data, list group_locs):
    cdef int i, j, k = 0
    cdef int nr = data.shape[0]
    cdef int nc = data.shape[1]
    cdef ndarray[np.uint8_t, ndim=2, cast=True] result = np.empty((size, nc - len(group_locs)), dtype='bool')
    for i in range(nc):
        if i in group_locs:
            k += 1
            continue
        for j in range(nr):
            result[labels[j], i - k] = data[j, i]
    return result

def last_str(ndarray[np.int64_t] labels, int size, ndarray[object, ndim=2] data, list group_locs):
    cdef int i, j, k = 0
    cdef int nr = data.shape[0]
    cdef int nc = data.shape[1]
    cdef ndarray[object, ndim=2] result = np.empty((size, nc - len(group_locs)), dtype='O')
    for i in range(nc):
        if i in group_locs:
            k += 1
            continue
        for j in range(nr):
            result[labels[j], i - k] = data[j, i]
    return result

def _get_first_non_nan(ndarray[np.int64_t] labels, int size, ndarray[np.float64_t, ndim=2] data, list group_locs):
    cdef int i, j, k = 0
    cdef int nr = data.shape[0]
    cdef int nc = data.shape[1]
    cdef ndarray[np.float64_t, ndim=2] result = np.empty((size, nc - len(group_locs)), dtype='float64')
    for i in range(nc):
        if i in group_locs:
            k += 1
            continue
        for j in range(nr):
            if isnan(data[j, i]):
                continue
            result[labels[j], i - k] = data[j, i]
    return result


def var_int(ndarray[np.int64_t] labels, int size, ndarray[np.int64_t, ndim=2] data, list group_locs,
            ndarray[np.int64_t] first_position, int ddof=1):
    cdef int i, j, k = 0
    cdef int nr = data.shape[0]
    cdef int nc = data.shape[1]
    cdef ndarray[np.float64_t, ndim=2] result = np.empty((size, nc - len(group_locs)), dtype='float64')
    cdef ndarray[np.int64_t, ndim=2] ct = np.zeros((size, nc - len(group_locs)), dtype='int64')
    cdef ndarray[np.int64_t, ndim=2] Ex = np.zeros((size, nc - len(group_locs)), dtype='int64')
    cdef ndarray[np.int64_t, ndim=2] Ex2 = np.zeros((size, nc - len(group_locs)), dtype='int64')
    cdef ndarray[np.int64_t, ndim=2] First = data[first_position, :]

    for i in range(nc):
        if i in group_locs:
            k += 1
            continue
        for j in range(nr):
            ct[labels[j], i - k] += 1
            Ex[labels[j], i - k] = Ex[labels[j], i - k]+  (data[j, i] - First[labels[j], i - k])
            Ex2[labels[j], i - k] = Ex2[labels[j], i - k] + (data[j, i] - First[labels[j], i - k]) ** 2

    with np.errstate(invalid='ignore'):
        result = (Ex2 - (Ex * Ex) / ct) / (ct - ddof)
    result[ct <= ddof] = nan
    return result

def var_float(ndarray[np.int64_t] labels, int size, ndarray[np.float64_t, ndim=2] data, list group_locs,
            ndarray[np.int64_t] first_position, int ddof=1):
    cdef int i, j, k = 0
    cdef int nr = data.shape[0]
    cdef int nc = data.shape[1]
    cdef ndarray[np.float64_t, ndim=2] result = np.empty((size, nc - len(group_locs)), dtype='float64')
    cdef ndarray[np.int64_t, ndim=2] ct = np.zeros((size, nc - len(group_locs)), dtype='int64')
    cdef ndarray[np.float64_t, ndim=2] Ex = np.zeros((size, nc - len(group_locs)), dtype='float64')
    cdef ndarray[np.float64_t, ndim=2] Ex2 = np.zeros((size, nc - len(group_locs)), dtype='float64')

    # There must be a faster way to initialize this. Could just get first non-na of each column
    cdef ndarray[np.float64_t, ndim=2] First = _get_first_non_nan(labels, size, data, group_locs)
    for i in range(nc):
        if i in group_locs:
            k += 1
            continue
        for j in range(nr):
            if isnan(data[j, i]):
                continue
            ct[labels[j], i - k] += 1
            Ex[labels[j], i - k] = Ex[labels[j], i - k] +  (data[j, i] - First[labels[j], i])
            Ex2[labels[j], i - k] = Ex2[labels[j], i - k] + (data[j, i] - First[labels[j], i]) ** 2

    with np.errstate(invalid='ignore'):
        result = (Ex2 - (Ex * Ex) / ct) / (ct - ddof)
    result[ct <= ddof] = nan
    return result

def var_bool(ndarray[np.int64_t] labels, int size, ndarray[np.uint8_t, ndim=2, cast=True] data, list group_locs,
            ndarray[np.int64_t] first_position, int ddof=1):
    cdef int i, j, k = 0
    cdef int nr = data.shape[0]
    cdef int nc = data.shape[1]
    cdef ndarray[np.float64_t, ndim=2] result = np.empty((size, nc - len(group_locs)), dtype='float64')
    cdef ndarray[np.int64_t, ndim=2] ct = np.zeros((size, nc - len(group_locs)), dtype='int64')
    cdef ndarray[np.int64_t, ndim=2] Ex = np.zeros((size, nc - len(group_locs)), dtype='int64')
    cdef ndarray[np.int64_t, ndim=2] Ex2 = np.zeros((size, nc - len(group_locs)), dtype='int64')
    cdef ndarray[np.uint8_t, ndim=2, cast=True] First = data[first_position, :]
    for i in range(nc):
        if i in group_locs:
            k += 1
            continue
        for j in range(nr):
            ct[labels[j], i - k] += 1
            Ex[labels[j], i - k] = Ex[labels[j], i - k]+  (data[j, i] - First[labels[j], i])
            Ex2[labels[j], i - k] = Ex2[labels[j], i - k] + (data[j, i] - First[labels[j], i]) ** 2

    with np.errstate(invalid='ignore'):
        result = (Ex2 - (Ex * Ex) / ct) / (ct - ddof)
    result[ct <= ddof] = nan
    return result


def cov_int(ndarray[np.int64_t] labels, int size, ndarray[np.int64_t, ndim=2] data, list group_locs):
    cdef int i, j, k = 0
    cdef int start, end
    cdef int nr = data.shape[0]
    cdef int nc = data.shape[1]
    cdef ndarray[np.float64_t, ndim=2] result = np.empty((size, nc - len(group_locs)), dtype='float64')
    cdef ndarray[np.int64_t] group_end_idx = np.empty(size, dtype='int64')
    cdef ndarray[np.int64_t]label_args = np.argsort(labels)
    cdef ndarray[np.int64_t] ordered_labels = labels[label_args]
    cdef ndarray[np.int64_t, ndim=2] data_sorted = data[label_args]
    cdef list covs
    cdef ndarray[np.int64_t, ndim=2] x
    cdef ndarray[np.int64_t] x0
    cdef ndarray[np.int64_t, ndim=2] x_diff
    cdef ndarray[np.int64_t, ndim=2] Ex
    cdef ndarray[np.int64_t, ndim=2] Exy
    cdef ndarray[np.int64_t, ndim=2] ExEy
    cdef int counts

    j = 0
    for i in range(1, nr):
        if ordered_labels[i - 1] != ordered_labels[i]:
            group_end_idx[j] = i
            j += 1
    group_end_idx[size - 1] = nr

    covs = []
    start = 0
    for i in range(size):
        end = group_end_idx[i]
        x = data_sorted[start:end]
        start = end

        x0 = x[0]
        x_diff = x - x0
        Exy = (x_diff.T @ x_diff)
        Ex = x_diff.sum(0)[np.newaxis, :]
        ExEy = Ex.T @ Ex
        counts = len(x)

        with np.errstate(invalid='ignore'):
            cov = (Exy - ExEy / counts) / (counts - 1)

        covs.append(cov)
    return np.row_stack(covs)

def cov_float(ndarray[np.int64_t] labels, int size, ndarray[np.float64_t, ndim=2] data, list group_locs):
    cdef int i, j, k = 0
    cdef int start, end
    cdef int nr = data.shape[0]
    cdef int nc = data.shape[1]
    cdef ndarray[np.float64_t, ndim=2] result = np.empty((size, nc - len(group_locs)), dtype='float64')
    cdef ndarray[np.int64_t] group_end_idx = np.empty(size, dtype='int64')
    cdef ndarray[np.int64_t]label_args = np.argsort(labels)
    cdef ndarray[np.int64_t] ordered_labels = labels[label_args]
    cdef ndarray[np.float64_t, ndim=2] data_sorted = data[label_args]
    cdef list covs
    cdef ndarray[np.float64_t, ndim=2] x
    cdef ndarray[np.float64_t] x0
    cdef ndarray[np.float64_t, ndim=2] x_diff
    cdef ndarray[np.float64_t, ndim=2] x_diff_0
    cdef ndarray[np.int64_t, ndim=2] x_not_nan
    cdef ndarray[np.float64_t, ndim=2] Ex
    cdef ndarray[np.float64_t, ndim=2] Exy
    cdef ndarray[np.float64_t, ndim=2] ExEy
    cdef ndarray[np.int64_t, ndim=2] counts

    j = 0
    for i in range(1, nr):
        if ordered_labels[i - 1] != ordered_labels[i]:
            group_end_idx[j] = i
            j += 1
    group_end_idx[size - 1] = nr

    covs = []
    start = 0
    for i in range(size):
        end = group_end_idx[i]
        x = data_sorted[start:end]
        start = end

        x0 = get_first_non_nan(x)
        x_diff = x - x0
        x_not_nan = (~np.isnan(x)).astype(int)

        x_diff_0 = np.nan_to_num(x_diff)
        counts = (x_not_nan.T @ x_not_nan)
        Exy = (x_diff_0.T @ x_diff_0)
        Ex = (x_diff_0.T @ x_not_nan)
        ExEy = Ex * Ex.T

        with np.errstate(invalid='ignore'):
            cov = (Exy - ExEy / counts) / (counts - 1)

        covs.append(cov)
    return np.row_stack(covs)

def corr_int(ndarray[np.int64_t] labels, int size, ndarray[np.int64_t, ndim=2] data, list group_locs):
    cdef int i, j, k = 0
    cdef int start, end
    cdef int nr = data.shape[0]
    cdef int nc = data.shape[1]
    cdef ndarray[np.float64_t, ndim=2] result = np.empty((size, nc - len(group_locs)), dtype='float64')
    cdef ndarray[np.int64_t] group_end_idx = np.empty(size, dtype='int64')
    cdef ndarray[np.int64_t]label_args = np.argsort(labels)
    cdef ndarray[np.int64_t] ordered_labels = labels[label_args]
    cdef ndarray[np.int64_t, ndim=2] data_sorted = data[label_args]
    cdef list corrs
    cdef ndarray[np.int64_t, ndim=2] x
    cdef ndarray[np.int64_t] x0
    cdef ndarray[np.int64_t, ndim=2] x_diff
    cdef ndarray[np.int64_t, ndim=2] Ex
    cdef ndarray[np.int64_t, ndim=2] Ex2
    cdef ndarray[np.int64_t, ndim=2] Exy
    cdef ndarray[np.int64_t, ndim=2] ExEy
    cdef ndarray[np.float64_t, ndim=2] stdx
    cdef ndarray[np.float64_t, ndim=2] stdy
    cdef ndarray[np.float64_t, ndim=2] corr
    cdef int counts

    j = 0
    for i in range(1, nr):
        if ordered_labels[i - 1] != ordered_labels[i]:
            group_end_idx[j] = i
            j += 1
    group_end_idx[size - 1] = nr

    corrs = []
    start = 0
    for i in range(size):
        end = group_end_idx[i]
        x = data_sorted[start:end]
        start = end

        x0 = x[0]
        x_diff = x - x0
        Exy = (x_diff.T @ x_diff)
        Ex = x_diff.sum(0)[np.newaxis, :]
        ExEy = Ex.T @ Ex
        counts = len(x)
        Ex2 = (x_diff ** 2).sum(0)

        with np.errstate(invalid='ignore'):
            cov = (Exy - ExEy / counts) / (counts - 1)
            stdx = (Ex2 - Ex ** 2 / counts) / (counts - 1)
            stdxy = stdx * stdx.T
            corr = cov / np.sqrt(stdxy)

        corrs.append(corr)
    return np.row_stack(corrs)

def corr_float(ndarray[np.int64_t] labels, int size, ndarray[np.float64_t, ndim=2] data, list group_locs):
    cdef int i, j, k = 0
    cdef int start, end
    cdef int nr = data.shape[0]
    cdef int nc = data.shape[1]
    cdef ndarray[np.float64_t, ndim=2] result = np.empty((size, nc - len(group_locs)), dtype='float64')
    cdef ndarray[np.int64_t] group_end_idx = np.empty(size, dtype='int64')
    cdef ndarray[np.int64_t]label_args = np.argsort(labels)
    cdef ndarray[np.int64_t] ordered_labels = labels[label_args]
    cdef ndarray[np.float64_t, ndim=2] data_sorted = data[label_args]
    cdef list corrs
    cdef ndarray[np.float64_t, ndim=2] x
    cdef ndarray[np.float64_t] x0
    cdef ndarray[np.float64_t, ndim=2] x_diff
    cdef ndarray[np.float64_t, ndim=2] x_diff_0
    cdef ndarray[np.int64_t, ndim=2] x_not_nan
    cdef ndarray[np.float64_t, ndim=2] Ex
    cdef ndarray[np.float64_t, ndim=2] Ex2
    cdef ndarray[np.float64_t, ndim=2] Exy
    cdef ndarray[np.float64_t, ndim=2] ExEy
    cdef ndarray[np.float64_t, ndim=2] stdx
    cdef ndarray[np.float64_t, ndim=2] stdy
    cdef ndarray[np.float64_t, ndim=2] corr
    cdef ndarray[np.int64_t, ndim=2] counts

    j = 0
    for i in range(1, nr):
        if ordered_labels[i - 1] != ordered_labels[i]:
            group_end_idx[j] = i
            j += 1
    group_end_idx[size - 1] = nr

    corrs = []
    start = 0
    for i in range(size):
        end = group_end_idx[i]
        x = data_sorted[start:end]
        start = end

        x0 = get_first_non_nan(x)
        x_diff = x - x0
        x_not_nan = (~np.isnan(x)).astype(int)

        x_diff_0 = np.nan_to_num(x_diff)
        counts = (x_not_nan.T @ x_not_nan)
        Exy = (x_diff_0.T @ x_diff_0)
        Ex = (x_diff_0.T @ x_not_nan)
        ExEy = Ex * Ex.T
        Ex2 = (x_diff_0.T ** 2 @ x_not_nan)

        with np.errstate(invalid='ignore'):
            cov = (Exy - ExEy / counts) / (counts - 1)
            stdx = (Ex2 - Ex ** 2 / counts) / (counts - 1)
            stdxy = stdx * stdx.T
            corr = cov / np.sqrt(stdxy)

        corrs.append(corr)
    return np.row_stack(corrs)  
    
def any_int(ndarray[np.int64_t] labels, int size, ndarray[np.int64_t, ndim=2] data, list group_locs):
    cdef int i, j, k = 0
    cdef int nr = data.shape[0]
    cdef int nc = data.shape[1]
    cdef ndarray[np.uint8_t, ndim=2, cast=True] result = np.zeros((size, nc - len(group_locs)), dtype='bool')
    for i in range(nc):
        if i in group_locs:
            k += 1
            continue
        for j in range(nr):
            if data[j, i] != 0:
                result[labels[j], i - k] = True
    return result

def any_float(ndarray[np.int64_t] labels, int size, ndarray[np.float64_t, ndim=2] data, list group_locs):
    cdef int i, j, k = 0
    cdef int nr = data.shape[0]
    cdef int nc = data.shape[1]
    cdef ndarray[np.uint8_t, ndim=2, cast=True] result = np.zeros((size, nc - len(group_locs)), dtype='bool')
    for i in range(nc):
        if i in group_locs:
            k += 1
            continue
        for j in range(nr):
            if not isnan(data[j, i]) and data[j, i] != 0:
                result[labels[j], i - k] = True
    return result

def any_bool(ndarray[np.int64_t] labels, int size, ndarray[np.uint8_t, ndim=2, cast=True] data, list group_locs):
    cdef int i, j, k = 0
    cdef int nr = data.shape[0]
    cdef int nc = data.shape[1]
    cdef ndarray[np.uint8_t, ndim=2, cast=True] result = np.zeros((size, nc - len(group_locs)), dtype='bool')
    for i in range(nc):
        if i in group_locs:
            k += 1
            continue
        for j in range(nr):
            if data[j, i] != False:
                result[labels[j], i - k] = True
    return result

def any_str(ndarray[np.int64_t] labels, int size, ndarray[object, ndim=2] data, list group_locs):
    cdef int i, j, k = 0
    cdef int nr = data.shape[0]
    cdef int nc = data.shape[1]
    cdef ndarray[np.uint8_t, ndim=2, cast=True] result = np.zeros((size, nc - len(group_locs)), dtype='bool')
    for i in range(nc):
        if i in group_locs:
            k += 1
            continue
        for j in range(nr):
            if data[j, i] is not None and data[j, i] != 0:
                result[labels[j], i - k] = True
    return result

def all_int(ndarray[np.int64_t] labels, int size, ndarray[np.int64_t, ndim=2] data, list group_locs):
    cdef int i, j, k = 0
    cdef int nr = data.shape[0]
    cdef int nc = data.shape[1]
    cdef ndarray[np.uint8_t, ndim=2, cast=True] result = np.ones((size, nc - len(group_locs)), dtype='bool')
    for i in range(nc):
        if i in group_locs:
            k += 1
            continue
        for j in range(nr):
            if data[j, i] == 0:
                result[labels[j], i - k] = False
    return result

def all_float(ndarray[np.int64_t] labels, int size, ndarray[np.float64_t, ndim=2] data, list group_locs):
    cdef int i, j, k = 0
    cdef int nr = data.shape[0]
    cdef int nc = data.shape[1]
    cdef ndarray[np.uint8_t, ndim=2, cast=True] result = np.ones((size, nc - len(group_locs)), dtype='bool')
    for i in range(nc):
        if i in group_locs:
            k += 1
            continue
        for j in range(nr):
            if isnan(data[j, i]) or data[j, i] == 0:
                result[labels[j], i - k] = False
    return result

def all_bool(ndarray[np.int64_t] labels, int size, ndarray[np.uint8_t, ndim=2, cast=True] data, list group_locs):
    cdef int i, j, k = 0
    cdef int nr = data.shape[0]
    cdef int nc = data.shape[1]
    cdef ndarray[np.uint8_t, ndim=2, cast=True] result = np.ones((size, nc - len(group_locs)), dtype='bool')
    for i in range(nc):
        if i in group_locs:
            k += 1
            continue
        for j in range(nr):
            if data[j, i] == False:
                result[labels[j], i - k] = False
    return result

def all_str(ndarray[np.int64_t] labels, int size, ndarray[object, ndim=2] data, list group_locs):
    cdef int i, j, k = 0
    cdef int nr = data.shape[0]
    cdef int nc = data.shape[1]
    cdef ndarray[np.uint8_t, ndim=2, cast=True] result = np.ones((size, nc - len(group_locs)), dtype='bool')
    for i in range(nc):
        if i in group_locs:
            k += 1
            continue
        for j in range(nr):
            if data[j, i] is None or data[j, i] == 0:
                result[labels[j], i - k] = False
    return result


def median_int(ndarray[np.int64_t] labels, int size, ndarray[np.int64_t, ndim=2] data, list group_locs):
    cdef int i, j, k = 0
    cdef int start, end
    cdef int nr = data.shape[0]
    cdef int nc = data.shape[1]
    cdef ndarray[np.float64_t, ndim=2] result = np.empty((size, nc - len(group_locs)), dtype='float64')
    cdef ndarray[np.int64_t] group_end_idx = np.empty(size, dtype='int64')
    cdef ndarray[np.int64_t]label_args = np.argsort(labels)
    cdef ndarray[np.int64_t] ordered_labels = labels[label_args]
    cdef ndarray[np.int64_t, ndim=2] data_sorted = data[label_args]
    cdef list medians = []

    j = 0
    for i in range(1, nr):
        if ordered_labels[i - 1] != ordered_labels[i]:
            group_end_idx[j] = i
            j += 1
    group_end_idx[size - 1] = nr

    start = 0
    for i in range(size):
        end = group_end_idx[i]
        x = data_sorted[start:end]
        start = end
        medians.append(np.median(x, 0))
    return np.row_stack(medians)

def median_float(ndarray[np.int64_t] labels, int size, ndarray[np.float64_t, ndim=2] data, list group_locs):
    cdef int i, j, k = 0
    cdef int start, end
    cdef int nr = data.shape[0]
    cdef int nc = data.shape[1]
    cdef ndarray[np.float64_t, ndim=2] result = np.empty((size, nc - len(group_locs)), dtype='float64')
    cdef ndarray[np.int64_t] group_end_idx = np.empty(size, dtype='int64')
    cdef ndarray[np.int64_t]label_args = np.argsort(labels)
    cdef ndarray[np.int64_t] ordered_labels = labels[label_args]
    cdef ndarray[np.float64_t, ndim=2] data_sorted = data[label_args]
    cdef list medians = []

    j = 0
    for i in range(1, nr):
        if ordered_labels[i - 1] != ordered_labels[i]:
            group_end_idx[j] = i
            j += 1
    group_end_idx[size - 1] = nr

    start = 0
    for i in range(size):
        end = group_end_idx[i]
        x = data_sorted[start:end]
        start = end
        medians.append(bn.nanmedian(x, 0))
    return np.row_stack(medians)


def median_bool(ndarray[np.int64_t] labels, int size, ndarray[np.uint8_t, ndim=2, cast=True] data, list group_locs):
    cdef int i, j, k = 0
    cdef int start, end
    cdef int nr = data.shape[0]
    cdef int nc = data.shape[1]
    cdef ndarray[np.float64_t, ndim=2] result = np.empty((size, nc - len(group_locs)), dtype='float64')
    cdef ndarray[np.int64_t] group_end_idx = np.empty(size, dtype='int64')
    cdef ndarray[np.int64_t]label_args = np.argsort(labels)
    cdef ndarray[np.int64_t] ordered_labels = labels[label_args]
    cdef ndarray[np.uint8_t, ndim=2] data_sorted = data[label_args]
    cdef list medians = []

    j = 0
    for i in range(1, nr):
        if ordered_labels[i - 1] != ordered_labels[i]:
            group_end_idx[j] = i
            j += 1
    group_end_idx[size - 1] = nr

    start = 0
    for i in range(size):
        end = group_end_idx[i]
        x = data_sorted[start:end]
        start = end
        medians.append(np.median(x, 0))
    return np.row_stack(medians)

def nunique_bool(ndarray[np.int64_t] labels, int size, ndarray[np.uint8_t, ndim=2, cast=True] data, list group_locs):
    cdef int i, j, k
    cdef int nr = data.shape[0]
    cdef int nc = data.shape[1]
    cdef int nc_final = nc - len(group_locs)
    cdef ndarray[np.int64_t, ndim=2] result = np.empty((size, nc_final), dtype='int64')
    cdef ndarray[np.int64_t] group_end_idx = np.empty(size, dtype='int64')
    cdef ndarray[np.int64_t]label_args = np.argsort(labels)
    cdef ndarray[np.int64_t] ordered_labels = labels[label_args]
    cdef ndarray[np.uint8_t, ndim=2, cast=True] data_sorted = data[label_args]
    cdef ndarray[object] uniques
    cdef np.uint8_t first

    j = 0
    for i in range(1, nr):
        if ordered_labels[i - 1] != ordered_labels[i]:
            group_end_idx[j] = i
            j += 1
    group_end_idx[size - 1] = nr

    start = 0
    for i in range(size):
        end = group_end_idx[i]
        uniques = np.empty(nc_final, dtype='O')
        for j in range(nc_final):
            uniques[j] = set()
        for k in range(nc_final):
            first = data_sorted[0, k]
            result[i, k] = 1
            for j in range(start, end):
                if data_sorted[j, k] != first:
                    result[i, k] = 2
                    break
        start = end
    return result

def nunique_str(ndarray[np.int64_t] labels, int size, ndarray[object, ndim=2] data, list group_locs):
    cdef int i, j, k
    cdef int nr = data.shape[0]
    cdef int nc = data.shape[1]
    cdef int nc_final = nc - len(group_locs)
    cdef ndarray[np.int64_t, ndim=2] result = np.empty((size, nc_final), dtype='int64')
    cdef ndarray[np.int64_t] group_end_idx = np.empty(size, dtype='int64')
    cdef ndarray[np.int64_t]label_args = np.argsort(labels)
    cdef ndarray[np.int64_t] ordered_labels = labels[label_args]
    cdef ndarray[object, ndim=2] data_sorted = data[label_args]
    cdef ndarray[object] uniques

    j = 0
    for i in range(1, nr):
        if ordered_labels[i - 1] != ordered_labels[i]:
            group_end_idx[j] = i
            j += 1
    group_end_idx[size - 1] = nr

    start = 0
    for i in range(size):
        end = group_end_idx[i]
        uniques = np.empty(nc_final, dtype='O')
        for j in range(nc_final):
            uniques[j] = set()
        for j in range(start, end):
            for k in range(nc_final):
                uniques[k].add(data_sorted[j, k])
        for j in range(nc_final):
            result[i, j] = len(uniques[j])
        start = end
    return result

def nunique_int(ndarray[np.int64_t] labels, int size, ndarray[np.int64_t, ndim=2] data, list group_locs):
    cdef int i, j, k
    cdef int nr = data.shape[0]
    cdef int nc = data.shape[1]
    cdef int nc_final = nc - len(group_locs)
    cdef ndarray[np.int64_t, ndim=2] result = np.empty((size, nc_final), dtype='int64')
    cdef ndarray[np.int64_t] group_end_idx = np.empty(size, dtype='int64')
    cdef ndarray[np.int64_t]label_args = np.argsort(labels)
    cdef ndarray[np.int64_t] ordered_labels = labels[label_args]
    cdef ndarray[np.int64_t, ndim=2] data_sorted = data[label_args]
    cdef ndarray[object] uniques

    j = 0
    for i in range(1, nr):
        if ordered_labels[i - 1] != ordered_labels[i]:
            group_end_idx[j] = i
            j += 1
    group_end_idx[size - 1] = nr

    start = 0
    for i in range(size):
        end = group_end_idx[i]
        uniques = np.empty(nc_final, dtype='O')
        for j in range(nc_final):
            uniques[j] = set()
        for j in range(start, end):
            for k in range(nc_final):
                uniques[k].add(data_sorted[j, k])
        for j in range(nc_final):
            result[i, j] = len(uniques[j])
        start = end
    return result

def nunique_float(ndarray[np.int64_t] labels, int size, ndarray[np.float64_t, ndim=2] data, list group_locs):
    cdef int i, j, k
    cdef int nr = data.shape[0]
    cdef int nc = data.shape[1]
    cdef int nc_final = nc - len(group_locs)
    cdef ndarray[np.int64_t, ndim=2] result = np.empty((size, nc_final), dtype='int64')
    cdef ndarray[np.int64_t] group_end_idx = np.empty(size, dtype='int64')
    cdef ndarray[np.int64_t]label_args = np.argsort(labels)
    cdef ndarray[np.int64_t] ordered_labels = labels[label_args]
    cdef ndarray[np.float64_t, ndim=2] data_sorted = data[label_args]
    cdef ndarray[object] uniques

    j = 0
    for i in range(1, nr):
        if ordered_labels[i - 1] != ordered_labels[i]:
            group_end_idx[j] = i
            j += 1
    group_end_idx[size - 1] = nr

    start = 0
    for i in range(size):
        end = group_end_idx[i]
        uniques = np.empty(nc_final, dtype='O')
        for j in range(nc_final):
            uniques[j] = set()
        for j in range(start, end):
            for k in range(nc_final):
                uniques[k].add(data_sorted[j, k])
        for j in range(nc_final):
            result[i, j] = len(uniques[j])
        start = end
    return result

def head(ndarray[np.int64_t] labels, int size, n):
    cdef int i
    cdef int nr = len(labels)
    cdef ndarray[np.int64_t] count = np.zeros(size, dtype='int64')
    cdef list final_locs = []
    for i in range(nr):
        count[labels[i]] += 1
        if count[labels[i]] <= n:
            final_locs.append(i)
    return final_locs

def tail(ndarray[np.int64_t] labels, int size, n):
    cdef int i
    cdef int nr = len(labels)
    cdef ndarray[np.int64_t] count = np.zeros(size, dtype='int64')
    cdef list final_locs = []
    for i in range(nr - 1, -1, -1):
        count[labels[i]] += 1
        if count[labels[i]] <= n:
            final_locs.append(i)
    return final_locs[::-1]
