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
from .math import min_max_int, min_max_int2
from libc.stdlib cimport malloc, free

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
    cdef ndarray[np.float64_t, ndim=2] result = np.full((size, nc - len(group_locs)),MIN_FLOAT, dtype='float64')
    for i in range(nc):
        if i in group_locs:
            k += 1
            continue
        for j in range(nr):
            if data[j, i] > result[labels[j], i - k]:
                result[labels[j], i - k] = data[j, i]
    result[result == MIN_FLOAT] = nan
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
    cdef ndarray[object, ndim=2] result = np.zeros((size, nc - len(group_locs)), dtype='U').astype('O')
    for i in range(nc):
        if i in group_locs:
            k += 1
            continue
        for j in range(nr):
            if data[j, i] is not None:
                if data[j, i] > result[labels[j], i - k]:
                    result[labels[j], i - k] = data[j, i]
    # run my isna to change to None
    return result