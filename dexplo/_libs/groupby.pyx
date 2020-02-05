#cython: boundscheck=False
#cython: wraparound=False
import numpy as np
cimport numpy as np
from numpy cimport ndarray
import cython
from cpython cimport set, list, tuple
from libc.math cimport isnan, sqrt

from numpy import nan
from .math import min_max_int, min_max_int2, get_first_non_nan, quick_select_int2, quick_select_float2
from libc.stdlib cimport malloc, free
from cpython.bytes cimport PyBytes_FromStringAndSize
from cpython cimport dict
from dexplo import _utils
from collections import defaultdict

try:
    import bottleneck as bn
except ImportError:
    import numpy as bn

cdef np.float64_t MAX_FLOAT = np.finfo(np.float64).max
cdef np.float64_t MIN_FLOAT = np.finfo(np.float64).min

cdef np.int64_t MAX_INT = np.iinfo(np.int64).max
cdef np.int64_t MIN_INT = np.iinfo(np.int64).min

MAX_CHAR = chr(1_000_000)
MIN_CHAR = chr(0)


cdef extern from "numpy/npy_math.h":
    bint npy_isnan(double x)


def get_group_assignment_str_1d(ndarray[object] a):
    cdef:
        Py_ssize_t i
        int nr = a.shape[0], count = 0
        ndarray[np.int64_t] group = np.empty(nr, dtype=np.int64)
        ndarray[np.int64_t] group_position = np.empty(nr, dtype=np.int64)
        dict d = {}

    for i in range(nr):
        group[i] = d.get(a[i], -1)
        if group[i] == -1:
            group_position[count] = i
            group[i] = count
            d[a[i]] = count
            count += 1

    return group, group_position[:count]

def get_group_assignment_str_1d_idx(ndarray[object] a):
    cdef:
        Py_ssize_t i, j, k
        int group_num, nr = a.shape[0], nc = a.shape[1], count = 0
        ndarray[np.int64_t] group = np.empty(nr, dtype=np.int64)
        ndarray[np.int64_t] group_position = np.empty(nr, dtype=np.int64)
        ndarray[object] group_idx = np.empty(nr, dtype='O')
        dict d = {}

    for i in range(nr):
        group_idx[i] = []

    for i in range(nr):
        group_num = d.get(a[i], -1)
        if group_num == -1:
            group_position[count] = i
            group[i] = count
            d[a[i]] = count
            count += 1
        else:
            group[i] = group_num
            group_idx[group_num].append(i)

    return group, group_position[:count], group_idx[:count]

def get_group_assignment_str_2d(ndarray[object, ndim=2] a):
    cdef:
        Py_ssize_t i, j, k
        int nr = a.shape[0], nc = a.shape[1], count = 0
        ndarray[np.int64_t] group = np.empty(nr, dtype=np.int64)
        ndarray[np.int64_t] group_position = np.empty(nr, dtype=np.int64)
        dict d = {}
        list v = list(range(nc))
        tuple t

    for i in range(nr):
        for j in range(nc):
            v[j] = a[i, j]
        t = tuple(v)

        group[i] = d.get(t, -1)
        if group[i] == -1:
            group_position[count] = i
            group[i] = count
            d[t] = count
            count += 1

    return group, group_position[:count]

def get_group_assignment_int_1d(ndarray[np.int64_t] a):
    cdef:
        Py_ssize_t i
        int n = len(a), count = 0
        ndarray[np.int64_t] group = np.empty(n, dtype=np.int64)
        ndarray[np.int64_t] group_position = np.empty(n, dtype=np.int64)
        dict d = {}

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
    cdef:
        Py_ssize_t i
        int n = len(a), count = 0
        ndarray[np.int64_t] unique
        np.int64_t rng = high - low + 1
        ndarray[np.int64_t] group = np.empty(n, dtype=np.int64)
        ndarray[np.int64_t] group_position = np.empty(n, dtype=np.int64)

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
    cdef:
        Py_ssize_t i
        int n = len(a), nc = a.shape[1], count = 0
        ndarray[np.int64_t] group = np.empty(n, dtype=np.int64)
        ndarray[np.int64_t] group_position = np.empty(n, dtype=np.int64)
        dict d = {}
        ndarray[np.int64_t] ranges
        np.int64_t total_range = 1
        int cur_range = 10 ** 7, size = sizeof(np.int64_t) * nc
        bytes string

    lows, highs = min_max_int2(a, 0)

    ranges = highs - lows + 1
    for i in range(nc):
        cur_range /= ranges[i]
        total_range *= ranges[i]

    if cur_range > 1:
        return get_group_assignment_int_bounded_2d(a, lows, highs, ranges, total_range)

    for i in range(n):
        string = PyBytes_FromStringAndSize(<char*>&a[i, 0], size)
        group[i] = d.get(string, -1)
        if group[i] == -1:
            group_position[count] = i
            group[i] = count
            d[string] = count
            count += 1

    return group, group_position[:count]

def get_group_assignment_int_bounded_2d(ndarray[np.int64_t, ndim=2] a, ndarray[np.int64_t] lows,
                                        ndarray[np.int64_t] highs, ndarray[np.int64_t] ranges,
                                        int total_range):
    cdef:
        Py_ssize_t i, j
        int count = 0, n = len(a), nc = a.shape[1]
        ndarray[np.int64_t] group = np.empty(n, dtype=np.int64)
        ndarray[np.int64_t] group_position = np.empty(n, dtype=np.int64)
        ndarray[np.int64_t, ndim=2] idx
        long *unique = <long *>malloc(total_range * sizeof(long))
        long iloc
        ndarray[np.int64_t] range_prod

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
    cdef str s
    cdef int size = sizeof(np.float64_t) * nc

    for i in range(n):
        string = PyBytes_FromStringAndSize(<char*>&a[i, 0], size)
        group[i] = d.get(string, -1)
        if group[i] == -1:
            group_position[count] = i
            group[i] = count
            d[string] = count
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

def get_group_assignment_str_1d_int_1d(ndarray[object] a, ndarray[np.int64_t] b):
    cdef int i
    cdef int nr = len(a)
    cdef int count = 0
    cdef ndarray[np.int64_t] group = np.empty(nr, dtype=np.int64)
    cdef ndarray[np.int64_t] group_position = np.empty(nr, dtype=np.int64)
    cdef dict d = {}
    cdef tuple t

    for i in range(nr):
        t = (a[i], b[i])

        group[i] = d.get(t, -1)
        if group[i] == -1:
            group_position[count] = i
            group[i] = count
            d[t] = count
            count += 1

    return group, group_position[:count]

def get_group_assignment_str_1d_float_1d(ndarray[object] a, ndarray[np.float64_t] b):
    cdef int i
    cdef int nr = len(a)
    cdef int count = 0
    cdef ndarray[np.int64_t] group = np.empty(nr, dtype=np.int64)
    cdef ndarray[np.int64_t] group_position = np.empty(nr, dtype=np.int64)
    cdef dict d = {}
    cdef tuple t

    for i in range(nr):
        if isnan(b[i]):
            t = (a[i], None)
        else:
            t = (a[i], b[i])

        group[i] = d.get(t, -1)
        if group[i] == -1:
            group_position[count] = i
            group[i] = count
            d[t] = count
            count += 1

    return group, group_position[:count]

def get_group_assignment_str_1d_bool_1d(ndarray[object] a, ndarray[np.uint8_t, cast=True] b):
    cdef int i
    cdef int nr = len(a)
    cdef int count = 0
    cdef ndarray[np.int64_t] group = np.empty(nr, dtype=np.int64)
    cdef ndarray[np.int64_t] group_position = np.empty(nr, dtype=np.int64)
    cdef dict d = {}
    cdef tuple t

    for i in range(nr):
        t = (a[i], b[i])

        group[i] = d.get(t, -1)
        if group[i] == -1:
            group_position[count] = i
            group[i] = count
            d[t] = count
            count += 1

    return group, group_position[:count]

def get_group_assignment_str_2d_int_1d(ndarray[object, ndim=2] a, ndarray[np.int64_t] b):
    cdef int i, j
    cdef int nr = len(a)
    cdef int nc = a.shape[1]
    cdef int count = 0
    cdef ndarray[np.int64_t] group = np.empty(nr, dtype=np.int64)
    cdef ndarray[np.int64_t] group_position = np.empty(nr, dtype=np.int64)
    cdef dict d = {}
    cdef list v = list(range(nc + 1))
    cdef tuple t

    for i in range(nr):
        for j in range(nc):
            v[j] = a[i, j]
        v[nc] = b[i]
        t = tuple(v)

        group[i] = d.get(t, -1)
        if group[i] == -1:
            group_position[count] = i
            group[i] = count
            d[t] = count
            count += 1

    return group, group_position[:count]

def get_group_assignment_str_2d_float_1d(ndarray[object, ndim=2] a, ndarray[np.float64_t] b):
    cdef int i, j
    cdef int nr = len(a)
    cdef int nc = a.shape[1]
    cdef int count = 0
    cdef ndarray[np.int64_t] group = np.empty(nr, dtype=np.int64)
    cdef ndarray[np.int64_t] group_position = np.empty(nr, dtype=np.int64)
    cdef dict d = {}
    cdef list v = list(range(nc + 1))
    cdef tuple t

    for i in range(nr):
        for j in range(nc):
            v[j] = a[i, j]
        if isnan(b[i]):
            v[nc] = None
        else:
            v[nc] = b[i]
        t = tuple(v)

        group[i] = d.get(t, -1)
        if group[i] == -1:
            group_position[count] = i
            group[i] = count
            d[t] = count
            count += 1

    return group, group_position[:count]

def get_group_assignment_str_2d_bool_1d(ndarray[object, ndim=2] a, ndarray[np.uint8_t, cast=True] b):
    cdef int i, j
    cdef int nr = len(a)
    cdef int nc = a.shape[1]
    cdef int count = 0
    cdef ndarray[np.int64_t] group = np.empty(nr, dtype=np.int64)
    cdef ndarray[np.int64_t] group_position = np.empty(nr, dtype=np.int64)
    cdef dict d = {}
    cdef list v = list(range(nc + 1))
    cdef tuple t

    for i in range(nr):
        for j in range(nc):
            v[j] = a[i, j]
        v[nc] = b[i]
        t = tuple(v)

        group[i] = d.get(t, -1)
        if group[i] == -1:
            group_position[count] = i
            group[i] = count
            d[t] = count
            count += 1

    return group, group_position[:count]


def get_group_assignment_str_1d_int_2d(ndarray[object] a, ndarray[np.int64_t, ndim=2] b):
    cdef int i
    cdef int nr = len(a)
    cdef int count = 0
    cdef ndarray[np.int64_t] group = np.empty(nr, dtype=np.int64)
    cdef ndarray[np.int64_t] group_position = np.empty(nr, dtype=np.int64)
    cdef dict d = {}
    cdef tuple t
    cdef int size = sizeof(np.int64_t) * b.shape[1]

    for i in range(nr):
        t = (a[i], PyBytes_FromStringAndSize(<char*>&b[i, 0], size))

        group[i] = d.get(t, -1)
        if group[i] == -1:
            group_position[count] = i
            group[i] = count
            d[t] = count
            count += 1

    return group, group_position[:count]

def get_group_assignment_str_1d_float_2d(ndarray[object] a, ndarray[np.float64_t, ndim=2] b):
    cdef int i
    cdef int nr = len(a)
    cdef int count = 0
    cdef ndarray[np.int64_t] group = np.empty(nr, dtype=np.int64)
    cdef ndarray[np.int64_t] group_position = np.empty(nr, dtype=np.int64)
    cdef dict d = {}
    cdef tuple t
    cdef int size = sizeof(np.float64_t) * b.shape[1]

    for i in range(nr):
        t = (a[i], PyBytes_FromStringAndSize(<char*>&b[i, 0], size))
        group[i] = d.get(t, -1)
        if group[i] == -1:
            group_position[count] = i
            group[i] = count
            d[t] = count
            count += 1

    return group, group_position[:count]

def get_group_assignment_str_1d_bool_2d(ndarray[object] a, ndarray[np.uint8_t, ndim=2, cast=True] b):
    cdef int i
    cdef int nr = len(a)
    cdef int count = 0
    cdef ndarray[np.int64_t] group = np.empty(nr, dtype=np.int64)
    cdef ndarray[np.int64_t] group_position = np.empty(nr, dtype=np.int64)
    cdef dict d = {}
    cdef tuple t
    cdef int size = sizeof(np.uint8_t) * b.shape[1]

    for i in range(nr):
        t = (a[i], PyBytes_FromStringAndSize(<char*>&b[i, 0], size))
        group[i] = d.get(t, -1)
        if group[i] == -1:
            group_position[count] = i
            group[i] = count
            d[t] = count
            count += 1

    return group, group_position[:count]

def get_group_assignment_str_2d_int_2d(ndarray[object, ndim=2] a, ndarray[np.int64_t, ndim=2] b):
    cdef int i, j
    cdef int nr = len(a)
    cdef int nc = a.shape[1]
    cdef int count = 0
    cdef ndarray[np.int64_t] group = np.empty(nr, dtype=np.int64)
    cdef ndarray[np.int64_t] group_position = np.empty(nr, dtype=np.int64)
    cdef dict d = {}
    cdef list v = list(range(nc + 1))
    cdef tuple t
    cdef int size = sizeof(np.int64_t) * b.shape[1]

    for i in range(nr):
        for j in range(nc):
            v[j] = a[i, j]
        v[nc] = PyBytes_FromStringAndSize(<char*>&b[i, 0], size)
        t = tuple(v)

        group[i] = d.get(t, -1)
        if group[i] == -1:
            group_position[count] = i
            group[i] = count
            d[t] = count
            count += 1

    return group, group_position[:count]

def get_group_assignment_str_2d_float_2d(ndarray[object, ndim=2] a, ndarray[np.float64_t, ndim=2] b):
    cdef int i, j
    cdef int nr = len(a)
    cdef int nc = a.shape[1]
    cdef int count = 0
    cdef ndarray[np.int64_t] group = np.empty(nr, dtype=np.int64)
    cdef ndarray[np.int64_t] group_position = np.empty(nr, dtype=np.int64)
    cdef dict d = {}
    cdef list v = list(range(nc + 1))
    cdef tuple t
    cdef int size = sizeof(np.float64_t) * b.shape[1]

    for i in range(nr):
        for j in range(nc):
            v[j] = a[i, j]
        v[nc] = PyBytes_FromStringAndSize(<char*>&b[i, 0], size)
        t = tuple(v)

        group[i] = d.get(t, -1)
        if group[i] == -1:
            group_position[count] = i
            group[i] = count
            d[t] = count
            count += 1

    return group, group_position[:count]

def get_group_assignment_str_2d_bool_2d(ndarray[object, ndim=2] a, ndarray[np.uint8_t, ndim=2, cast=True] b):
    cdef int i, j
    cdef int nr = len(a)
    cdef int nc = a.shape[1]
    cdef int count = 0
    cdef ndarray[np.int64_t] group = np.empty(nr, dtype=np.int64)
    cdef ndarray[np.int64_t] group_position = np.empty(nr, dtype=np.int64)
    cdef dict d = {}
    cdef list v = list(range(nc + 1))
    cdef tuple t
    cdef int size = sizeof(np.uint8_t) * b.shape[1]

    for i in range(nr):
        for j in range(nc):
            v[j] = a[i, j]
        v[nc] = PyBytes_FromStringAndSize(<char*>&b[i, 0], size)
        t = tuple(v)

        group[i] = d.get(t, -1)
        if group[i] == -1:
            group_position[count] = i
            group[i] = count
            d[t] = count
            count += 1

    return group, group_position[:count]

def value_counts_int(ndarray[np.int64_t] a):
    cdef int i, group_num
    cdef int n = len(a)
    cdef int count = 0
    cdef ndarray[np.int64_t] group_name = np.empty(n, dtype=np.int64)
    cdef ndarray[np.int64_t] counts = np.empty(n, dtype=np.int64)
    cdef dict d = {}

    low, high = min_max_int(a)
    if high - low < 10_000_000:
        return value_counts_int_bounded(a, low, high)

    for i in range(n):
        group_num = d.get(a[i], -1)
        if group_num == -1:
            counts[count] = 1
            group_name[count] = i
            d[a[i]] = count
            count += 1
        else:
            counts[group_num] += 1
    return group_name[:count], counts[:count]

def value_counts_int_bounded(ndarray[np.int64_t] a, int low, int high):
    cdef int i
    cdef int n = len(a)

    cdef ndarray[np.int64_t] counts = np.zeros(high - low + 1, dtype=np.int64)

    for i in range(n):
        counts[a[i] - low] += 1
    nz = counts.nonzero()[0]
    return nz + low, counts[nz]

def value_counts_str(ndarray[object] a, dropna):
    cdef int i, j, k, group_num
    cdef int nr = a.shape[0]
    cdef int nc = a.shape[1]
    cdef int count = 0
    cdef ndarray[np.int64_t] group_position = np.empty(nr, dtype=np.int64)
    cdef ndarray[np.int64_t] group_count = np.zeros(nr, dtype=np.int64)
    cdef dict d = {}

    if dropna:
        for i in range(nr):
            if a[i] is None:
                continue
            group_num = d.get(a[i], -1)
            if group_num == -1:
                group_position[count] = i
                group_count[count] = 1
                d[a[i]] = count
                count += 1
            else:
                group_count[group_num] += 1
    else:
        for i in range(nr):
            group_num = d.get(a[i], -1)
            if group_num == -1:
                group_position[count] = i
                group_count[count] = 1
                d[a[i]] = count
                count += 1
            else:
                group_count[group_num] += 1

    return group_position[:count], group_count[:count]

def value_counts_float(ndarray[np.float64_t] a, dropna):
    cdef int i, group_num
    cdef int n = len(a)
    cdef int count = 0
    cdef ndarray[np.int64_t] group_name = np.empty(n, dtype='int64')
    cdef ndarray[np.int64_t] counts = np.empty(n, dtype='int64')
    cdef dict d = {}

    if dropna:
        for i in range(n):
            if isnan(a[i]):
                continue
            group_num = d.get(a[i], -1)
            if group_num == -1:
                counts[count] = 1
                group_name[count] = i
                d[a[i]] = count
                count += 1
            else:
                counts[group_num] += 1
    else:
        for i in range(n):
            if isnan(a[i]):
                v = nan
            else:
                v = a[i]
            group_num = d.get(v, -1)
            if group_num == -1:
                counts[count] = 1
                group_name[count] = i
                d[v] = count
                count += 1
            else:
                counts[group_num] += 1
    return group_name[:count], counts[:count]

def value_counts_bool(ndarray[np.uint8_t, cast=True] a):
    cdef int i, group_num
    cdef int n = len(a)
    cdef int count = 0
    cdef ndarray[np.uint8_t, cast=True] group_name = np.array([False, True])
    cdef ndarray[np.int64_t] counts = np.zeros(2, dtype=np.int64)
    cdef dict d = {}

    for i in range(n):
        counts[a[i]] += 1
    return group_name, counts

def size(ndarray[np.int64_t] a, int group_size):
    cdef int i
    cdef int n = len(a)
    cdef ndarray[np.int64_t] result = np.zeros(group_size, dtype=np.int64)
    for i in range(n):
        result[a[i]] += 1
    return result

def size_int(ndarray[np.int64_t] labels, int size, ndarray[np.int64_t, ndim=2] data, list group_locs):
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

def size_bool(ndarray[np.int64_t] labels, int size, ndarray[np.uint8_t, ndim=2, cast=True] data, list group_locs):
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

def size_float(ndarray[np.int64_t] labels, int size, ndarray[np.float64_t, ndim=2] data, list group_locs):
    cdef int i, j, k = 0
    cdef int nr = data.shape[0]
    cdef int nc = data.shape[1]
    cdef ndarray[np.int64_t, ndim=2] result = np.zeros((size, nc - len(group_locs)), dtype='int64')
    for i in range(nc):
        if i in group_locs:
            k += 1
            continue
        for j in range(nr):
            result[labels[j], i - k] += 1
    return result

def size_str(ndarray[np.int64_t] labels, int size, ndarray[object, ndim=2] data, list group_locs):
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

def count_date(ndarray[np.int64_t] labels, int size, ndarray[np.int64_t, ndim=2] data, list group_locs):
    cdef int i, j
    cdef int nr = data.shape[0]
    cdef int nc = data.shape[1]
    cdef ndarray[np.int64_t, ndim=2] result = np.zeros((size, nc - len(group_locs)), dtype='int64')
    cdef int k = 0
    cdef long nat = np.datetime64('nat').astype('int64')
    for i in range(nc):
        if i in group_locs:
            k += 1
            continue
        for j in range(nr):
            if data[j, i] != nat:
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

def prod_int(ndarray[np.int64_t] labels, int size, ndarray[np.int64_t, ndim=2] data, list group_locs):
    cdef int i, j, k = 0
    cdef int nr = data.shape[0]
    cdef int nc = data.shape[1]
    cdef ndarray[np.int64_t, ndim=2] result = np.ones((size, nc - len(group_locs)), dtype='int64')
    for i in range(nc):
        if i in group_locs:
            k += 1
            continue
        for j in range(nr):
            result[labels[j], i - k] = result[labels[j], i - k] * data[j, i]
    return result

def prod_float(ndarray[np.int64_t] labels, int size, ndarray[np.float64_t, ndim=2] data, list group_locs):
    cdef int i, j, k = 0
    cdef int nr = data.shape[0]
    cdef int nc = data.shape[1]
    cdef ndarray[np.float64_t, ndim=2] result = np.ones((size, nc - len(group_locs)), dtype='float64')
    for i in range(nc):
        if i in group_locs:
            k += 1
            continue
        for j in range(nr):
            if not isnan(data[j, i]):
                result[labels[j], i - k] = result[labels[j], i - k] * data[j, i]
    return result

def prod_bool(ndarray[np.int64_t] labels, int size, ndarray[np.uint8_t, ndim=2, cast=True] data, list group_locs):
    cdef int i, j, k = 0
    cdef int nr = data.shape[0]
    cdef int nc = data.shape[1]
    cdef ndarray[np.int64_t, ndim=2] result = np.ones((size, nc - len(group_locs)), dtype='int64')
    for i in range(nc):
        if i in group_locs:
            k += 1
            continue
        for j in range(nr):
            result[labels[j], i - k] = result[labels[j], i - k] * data[j, i]
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

def max_date(ndarray[np.int64_t] labels, int size, ndarray[np.int64_t, ndim=2] data, list group_locs):
    cdef int i, j, k = 0
    cdef int nr = data.shape[0]
    cdef int nc = data.shape[1]
    cdef ndarray[np.int64_t, ndim=2] result = np.full((size, nc - len(group_locs)), MIN_INT, dtype='int64')
    cdef long nat = np.datetime64('nat').astype('int64')
    for i in range(nc):
        if i in group_locs:
            k += 1
            continue
        for j in range(nr):
            if data[j, i] > result[labels[j], i - k] and data[j, i] != nat:
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

def min_date(ndarray[np.int64_t] labels, int size, ndarray[np.int64_t, ndim=2] data, list group_locs):
    cdef int i, j, k = 0
    cdef int nr = data.shape[0]
    cdef int nc = data.shape[1]
    cdef long nat = np.datetime64('nat').astype('int64')
    cdef ndarray[np.int64_t, ndim=2] result = np.full((size, nc - len(group_locs)), nat, dtype='int64')

    for i in range(nc):
        if i in group_locs:
            k += 1
            continue
        for j in range(nr):
            if (data[j, i] < result[labels[j], i - k] and data[j, i] != nat) or (result[labels[j], i - k] == nat):
                result[labels[j], i - k] = data[j, i]
    return result

def first_int(ndarray[np.int64_t] labels, int size, ndarray[np.int64_t, ndim=2] data, list group_locs):
    cdef int i
    cdef int nc = data.shape[1]
    cdef list kept_cols = []
    cdef list kept_rows = []

    for i in range(nc):
        if i not in group_locs:
            kept_cols.append(i)

    for i in range(size):
        kept_rows.append(labels[i])

    return data[np.ix_(kept_rows, kept_cols)]

def first_float(ndarray[np.int64_t] labels, int size, ndarray[np.float64_t, ndim=2] data, list group_locs):
    cdef int i
    cdef int nc = data.shape[1]
    cdef list kept_cols = []
    cdef list kept_rows = []

    for i in range(nc):
        if i not in group_locs:
            kept_cols.append(i)

    for i in range(size):
        kept_rows.append(labels[i])

    return data[np.ix_(kept_rows, kept_cols)]

def first_bool(ndarray[np.int64_t] labels, int size, ndarray[np.uint8_t, ndim=2, cast=True] data, list group_locs):
    cdef int i
    cdef int nc = data.shape[1]
    cdef list kept_cols = []
    cdef list kept_rows = []

    for i in range(nc):
        if i not in group_locs:
            kept_cols.append(i)

    for i in range(size):
        kept_rows.append(labels[i])

    return data[np.ix_(kept_rows, kept_cols)]

def first_str(ndarray[np.int64_t] labels, int size, ndarray[object, ndim=2] data, list group_locs):
    cdef int i
    cdef int nc = data.shape[1]
    cdef list kept_cols = []
    cdef list kept_rows = []

    for i in range(nc):
        if i not in group_locs:
            kept_cols.append(i)

    for i in range(size):
        kept_rows.append(labels[i])

    return data[np.ix_(kept_rows, kept_cols)]

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

def any_date(ndarray[np.int64_t] labels, int size, ndarray[np.int64_t, ndim=2] data, list group_locs):
    cdef int i, j, k = 0
    cdef int nr = data.shape[0]
    cdef int nc = data.shape[1]
    cdef ndarray[np.uint8_t, ndim=2, cast=True] result = np.zeros((size, nc - len(group_locs)), dtype='bool')
    cdef long nat = np.datetime64('nat').astype('int64')

    for i in range(nc):
        if i in group_locs:
            k += 1
            continue
        for j in range(nr):
            if data[j, i] != nat:
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

def all_date(ndarray[np.int64_t] labels, int size, ndarray[np.int64_t, ndim=2] data, list group_locs):
    cdef int i, j, k = 0
    cdef int nr = data.shape[0]
    cdef int nc = data.shape[1]
    cdef ndarray[np.uint8_t, ndim=2, cast=True] result = np.ones((size, nc - len(group_locs)), dtype='bool')
    cdef long nat = np.datetime64('nat').astype('int64')

    for i in range(nc):
        if i in group_locs:
            k += 1
            continue
        for j in range(nr):
            if data[j, i] == nat:
                result[labels[j], i - k] = False
    return result

def median_int(ndarray[np.int64_t] labels, int size, ndarray[np.int64_t, ndim=2] data, list group_locs):
    cdef int i, j, k = 0
    cdef long start, end, xlen, med_idx
    cdef np.float64_t first, second
    cdef int nr = data.shape[0]
    cdef int nc = data.shape[1]
    cdef ndarray[np.float64_t, ndim=2] result = np.empty((size, nc - len(group_locs)), dtype='float64')
    cdef ndarray[np.int64_t] label_count = np.zeros(size, dtype='int64')
    cdef ndarray[np.int64_t] label_cumsum = np.empty(size, dtype='int64')
    cdef ndarray[np.int64_t] label_locs = np.empty(nr, dtype='int64')
    cdef ndarray[np.int64_t] label_count_cur = np.zeros(size, dtype='int64')
    cdef ndarray[np.int64_t] x

    for i in range(nr):
        label_count[labels[i]] += 1

    label_cumsum = np.roll(label_count, 1)
    label_cumsum[0] = 0
    label_cumsum = label_cumsum.cumsum()

    for i in range(nr):
        label_locs[label_cumsum[labels[i]] + label_count_cur[labels[i]]] = i
        label_count_cur[labels[i]] += 1

    start = 0
    for i in range(size):
        end = start + label_count[i]
        k = 0
        for j in range(nc):
            if j in group_locs:
                k += 1
                continue
            x = data[:, j][label_locs[start:end]]
            x_len = x.shape[0]
            med_idx = x_len // 2
            if x_len % 2 == 1:
                result[i, j - k] = quick_select_int2(x, x_len, med_idx)
            else:
                first = quick_select_int2(x, x_len, med_idx - 1)
                second = quick_select_int2(x, x_len, med_idx)
                result[i, j - k] = (first + second) / 2

        start = end
    return result

def median_float(ndarray[np.int64_t] labels, int size, ndarray[np.float64_t, ndim=2] data, list group_locs):
    cdef int i, j, k = 0
    cdef long start, end, xlen, med_idx
    cdef np.float64_t first, second
    cdef int nr = data.shape[0]
    cdef int nc = data.shape[1]
    cdef ndarray[np.float64_t, ndim=2] result = np.empty((size, nc - len(group_locs)), dtype='float64')
    cdef ndarray[np.int64_t] label_count = np.zeros(size, dtype='int64')
    cdef ndarray[np.int64_t] label_cumsum = np.empty(size, dtype='int64')
    cdef ndarray[np.int64_t] label_locs = np.empty(nr, dtype='int64')
    cdef ndarray[np.int64_t] label_count_cur = np.zeros(size, dtype='int64')
    cdef ndarray[np.float64_t] x

    for i in range(nr):
        label_count[labels[i]] += 1

    label_cumsum = np.roll(label_count, 1)
    label_cumsum[0] = 0
    label_cumsum = label_cumsum.cumsum()

    for i in range(nr):
        label_locs[label_cumsum[labels[i]] + label_count_cur[labels[i]]] = i
        label_count_cur[labels[i]] += 1

    start = 0
    for i in range(size):
        end = start + label_count[i]
        k = 0
        for j in range(nc):
            if j in group_locs:
                k += 1
                continue
            x = data[:, j][label_locs[start:end]]
            x = x[~np.isnan(x)]
            x_len = x.shape[0]
            med_idx = x_len // 2
            if x_len == 0:
                result[i, j - k] = nan
            else:
                if x_len % 2 == 1:
                    result[i, j - k] = quick_select_float2(x, x_len, med_idx)
                else:
                    first = quick_select_float2(x, x_len, med_idx - 1)
                    second = quick_select_float2(x, x_len, med_idx)
                    result[i, j - k] = (first + second) / 2

        start = end
    return result

def median_bool(ndarray[np.int64_t] labels, int size, ndarray[np.uint8_t, ndim=2, cast=True] data, list group_locs):
    cdef int i, j, k = 0
    cdef long start, end, xlen, med_idx
    cdef np.float64_t first, second
    cdef int nr = data.shape[0]
    cdef int nc = data.shape[1]
    cdef ndarray[np.float64_t, ndim=2] result = np.empty((size, nc - len(group_locs)), dtype='float64')
    cdef ndarray[np.int64_t] label_count = np.zeros(size, dtype='int64')
    cdef ndarray[np.int64_t] label_cumsum = np.empty(size, dtype='int64')
    cdef ndarray[np.int64_t] label_locs = np.empty(nr, dtype='int64')
    cdef ndarray[np.int64_t] label_count_cur = np.zeros(size, dtype='int64')
    cdef ndarray[np.int64_t] x

    for i in range(nr):
        label_count[labels[i]] += 1

    label_cumsum = np.roll(label_count, 1)
    label_cumsum[0] = 0
    label_cumsum = label_cumsum.cumsum()

    for i in range(nr):
        label_locs[label_cumsum[labels[i]] + label_count_cur[labels[i]]] = i
        label_count_cur[labels[i]] += 1

    start = 0
    for i in range(size):
        end = start + label_count[i]
        k = 0
        for j in range(nc):
            if j in group_locs:
                k += 1
                continue
            x = data[:, j][label_locs[start:end]].astype('int64')
            x_len = x.shape[0]
            med_idx = x_len // 2
            if x_len % 2 == 1:
                result[i, j - k] = quick_select_int2(x, end-start, med_idx)
            else:
                first = quick_select_int2(x, end-start, med_idx - 1)
                second = quick_select_int2(x, end-start, med_idx)
                result[i, j - k] = (first + second) / 2

        start = end
    return result

def nunique_bool(ndarray[np.int64_t] labels, int size, ndarray[np.uint8_t, ndim=2, cast=True] data, list group_locs):
    cdef int i, j, k, g
    cdef int nr = data.shape[0]
    cdef int nc = data.shape[1]
    cdef int nc_final = nc - len(group_locs)
    cdef ndarray[np.int64_t, ndim=2] result = np.empty((size, nc_final), dtype='int64')
    cdef ndarray[np.int64_t] group_end_idx = np.empty(size, dtype='int64')
    cdef ndarray[np.int64_t] label_args = np.argsort(labels)
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
        g = 0
        for j in range(nc_final):
            uniques[j] = set()
        for k in range(nc):
            if k in group_locs:
                g += 1
                continue
            first = data_sorted[0, k]
            result[i, k - g] = 1
            for j in range(start, end):
                if data_sorted[j, k] != first:
                    result[i, k - g] = 2
                    break
        start = end
    return result

def nunique_str(ndarray[np.int64_t] labels, int size, ndarray[object, ndim=2] data, list group_locs):
    cdef int i, j, k, g
    cdef int nr = data.shape[0]
    cdef int nc = data.shape[1]
    cdef int nc_final = nc - len(group_locs)
    cdef ndarray[np.int64_t, ndim=2] result = np.empty((size, nc_final), dtype='int64')
    cdef ndarray[np.int64_t] group_end_idx = np.empty(size, dtype='int64')
    cdef ndarray[np.int64_t] label_args = np.argsort(labels)
    cdef ndarray[np.int64_t] ordered_labels = labels[label_args]
    cdef ndarray[object, ndim=2] data_sorted = data[label_args]
    cdef ndarray[object] uniques
    cdef set group_locs_set = set(group_locs)

    j = 0
    for i in range(1, nr):
        if ordered_labels[i - 1] != ordered_labels[i]:
            group_end_idx[j] = i
            j += 1
    group_end_idx[size - 1] = nr

    start = 0
    for i in range(size):
        end = group_end_idx[i]
        g = 0
        for k in range(nc):
            if k in group_locs_set:
                g += 1
                continue
            uniques2 = set()
            for j in range(start, end):
                uniques2.add(data_sorted[j, k])

            result[i, k - g] = len(uniques2)
        start = end

    return result

def nunique_int(ndarray[np.int64_t] labels, int size, ndarray[np.int64_t, ndim=2] data, list group_locs):
    cdef int i, j, k, g
    cdef int nr = data.shape[0]
    cdef int nc = data.shape[1]
    cdef int nc_final = nc - len(group_locs)
    cdef ndarray[np.int64_t, ndim=2] result = np.empty((size, nc_final), dtype='int64')
    cdef ndarray[np.int64_t] group_end_idx = np.empty(size, dtype='int64')
    cdef ndarray[np.int64_t]label_args = np.argsort(labels)
    cdef ndarray[np.int64_t] ordered_labels = labels[label_args]
    cdef ndarray[np.int64_t, ndim=2] data_sorted = data[label_args]
    cdef ndarray[object] uniques
    cdef set group_locs_set = set(group_locs)

    j = 0
    for i in range(1, nr):
        if ordered_labels[i - 1] != ordered_labels[i]:
            group_end_idx[j] = i
            j += 1
    group_end_idx[size - 1] = nr

    start = 0
    for i in range(size):
        end = group_end_idx[i]
        g = 0
        for k in range(nc):
            if k in group_locs_set:
                g += 1
                continue
            uniques2 = set()
            for j in range(start, end):
                uniques2.add(data_sorted[j, k])

            result[i, k - g] = len(uniques2)
        start = end
    return result

def nunique_float(ndarray[np.int64_t] labels, int size, ndarray[np.float64_t, ndim=2] data, list group_locs):
    cdef int i, j, k, g
    cdef int nr = data.shape[0]
    cdef int nc = data.shape[1]
    cdef int nc_final = nc - len(group_locs)
    cdef ndarray[np.int64_t, ndim=2] result = np.empty((size, nc_final), dtype='int64')
    cdef ndarray[np.int64_t] group_end_idx = np.empty(size, dtype='int64')
    cdef ndarray[np.int64_t]label_args = np.argsort(labels)
    cdef ndarray[np.int64_t] ordered_labels = labels[label_args]
    cdef ndarray[np.float64_t, ndim=2] data_sorted = data[label_args]
    cdef ndarray[object] uniques
    cdef set group_locs_set = set(group_locs)

    j = 0
    for i in range(1, nr):
        if ordered_labels[i - 1] != ordered_labels[i]:
            group_end_idx[j] = i
            j += 1
    group_end_idx[size - 1] = nr

    start = 0
    for i in range(size):
        end = group_end_idx[i]
        g = 0
        for k in range(nc):
            if k in group_locs_set:
                g += 1
                continue
            uniques2 = set()
            for j in range(start, end):
                uniques2.add(data_sorted[j, k])

            result[i, k - g] = len(uniques2)
        start = end
    return result

def nunique_date(ndarray[np.int64_t] labels, int size, ndarray[np.int64_t, ndim=2] data, list group_locs):
    cdef int i, j, k, g
    cdef int nr = data.shape[0]
    cdef int nc = data.shape[1]
    cdef int nc_final = nc - len(group_locs)
    cdef ndarray[np.int64_t, ndim=2] result = np.empty((size, nc_final), dtype='int64')
    cdef ndarray[np.int64_t] group_end_idx = np.empty(size, dtype='int64')
    cdef ndarray[np.int64_t]label_args = np.argsort(labels)
    cdef ndarray[np.int64_t] ordered_labels = labels[label_args]
    cdef ndarray[np.int64_t, ndim=2] data_sorted = data[label_args]
    cdef ndarray[object] uniques
    cdef set group_locs_set = set(group_locs)

    j = 0
    for i in range(1, nr):
        if ordered_labels[i - 1] != ordered_labels[i]:
            group_end_idx[j] = i
            j += 1
    group_end_idx[size - 1] = nr

    start = 0
    for i in range(size):
        end = group_end_idx[i]
        g = 0
        for k in range(nc):
            if k in group_locs_set:
                g += 1
                continue
            uniques2 = set()
            for j in range(start, end):
                uniques2.add(data_sorted[j, k])

            result[i, k - g] = len(uniques2)
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


def cummax_int(ndarray[np.int64_t] labels, int size, ndarray[np.int64_t, ndim=2] data, list group_locs):
    cdef int i, j, k = 0
    cdef int nr = data.shape[0]
    cdef int nc = data.shape[1]
    cdef int nc_final = nc - len(group_locs)
    cdef ndarray[np.int64_t, ndim=2] result = np.empty((nr, nc_final), dtype='int64')
    cdef ndarray[np.int64_t, ndim=2] cur_max = np.full((size, nc_final), MIN_INT, dtype='int64')
    for i in range(nc):
        if i in group_locs:
            k += 1
            continue
        for j in range(nr):
            if data[j, i] > cur_max[labels[j], i - k]:
                cur_max[labels[j], i - k] = data[j, i]
            result[j, i - k] = cur_max[labels[j], i - k]
    return result

def cummax_float(ndarray[np.int64_t] labels, int size, ndarray[np.float64_t, ndim=2] data, list group_locs):
    cdef int i, j, k = 0
    cdef int nr = data.shape[0]
    cdef int nc = data.shape[1]
    cdef int nc_final = nc - len(group_locs)
    cdef ndarray[np.float64_t, ndim=2] result = np.empty((nr, nc_final), dtype='float64')
    cdef ndarray[np.float64_t, ndim=2] cur_max = np.full((size, nc_final), nan, dtype='float64')
    for i in range(nc):
        if i in group_locs:
            k += 1
            continue
        for j in range(nr):
            if not isnan(data[j, i]) and (isnan(cur_max[labels[j], i - k]) or data[j, i] > cur_max[labels[j], i - k]):
                cur_max[labels[j], i - k] = data[j, i]
            result[j, i - k] = cur_max[labels[j], i - k]
    return result

def cummax_bool(ndarray[np.int64_t] labels, int size, ndarray[np.uint8_t, ndim=2, cast=True] data, list group_locs):
    cdef int i, j, k = 0
    cdef int nr = data.shape[0]
    cdef int nc = data.shape[1]
    cdef int nc_final = nc - len(group_locs)
    cdef ndarray[np.uint8_t, ndim=2, cast=True] result = np.empty((nr, nc_final), dtype='bool')
    cdef ndarray[np.uint8_t, ndim=2, cast=True] cur_max = np.full((size, nc_final), False, dtype='bool')
    for i in range(nc):
        if i in group_locs:
            k += 1
            continue
        for j in range(nr):
            if cur_max[labels[j], i - k] == True:
                result[j, i - k] = True
                continue
            elif data[j, i] == True:
                cur_max[labels[j], i - k] = True
                result[j, i - k] = True
            else:
                result[j, i - k] = False
    return result

def cummax_date(ndarray[np.int64_t] labels, int size, ndarray[np.int64_t, ndim=2] data, list group_locs):
    cdef int i, j, k = 0
    cdef int nr = data.shape[0]
    cdef int nc = data.shape[1]
    cdef int nc_final = nc - len(group_locs)
    cdef long nat = np.datetime64('nat').astype('int64')
    cdef ndarray[np.int64_t, ndim=2] result = np.empty((nr, nc_final), dtype='int64')
    cdef ndarray[np.int64_t, ndim=2] cur_max = np.full((size, nc_final), nat, dtype='int64')
    for i in range(nc):
        if i in group_locs:
            k += 1
            continue
        for j in range(nr):
            if data[j, i] > cur_max[labels[j], i - k] and data[j, i] != nat:
                cur_max[labels[j], i - k] = data[j, i]
            result[j, i - k] = cur_max[labels[j], i - k]
    return result

def cummin_int(ndarray[np.int64_t] labels, int size, ndarray[np.int64_t, ndim=2] data, list group_locs):
    cdef int i, j, k = 0
    cdef int nr = data.shape[0]
    cdef int nc = data.shape[1]
    cdef int nc_final = nc - len(group_locs)
    cdef ndarray[np.int64_t, ndim=2] result = np.empty((nr, nc_final), dtype='int64')
    cdef ndarray[np.int64_t, ndim=2] cur_min = np.full((size, nc_final), MAX_INT, dtype='int64')
    for i in range(nc):
        if i in group_locs:
            k += 1
            continue
        for j in range(nr):
            if data[j, i] < cur_min[labels[j], i - k]:
                cur_min[labels[j], i - k] = data[j, i]
            result[j, i - k] = cur_min[labels[j], i - k]
    return result

def cummin_float(ndarray[np.int64_t] labels, int size, ndarray[np.float64_t, ndim=2] data, list group_locs):
    cdef int i, j, k = 0
    cdef int nr = data.shape[0]
    cdef int nc = data.shape[1]
    cdef int nc_final = nc - len(group_locs)
    cdef ndarray[np.float64_t, ndim=2] result = np.empty((nr, nc_final), dtype='float64')
    cdef ndarray[np.float64_t, ndim=2] cur_min = np.full((size, nc_final), nan, dtype='float64')
    for i in range(nc):
        if i in group_locs:
            k += 1
            continue
        for j in range(nr):
            if not isnan(data[j, i]) and (isnan(cur_min[labels[j], i - k]) or data[j, i] < cur_min[labels[j], i - k]):
                cur_min[labels[j], i - k] = data[j, i]
            result[j, i - k] = cur_min[labels[j], i - k]
    return result

def cummin_bool(ndarray[np.int64_t] labels, int size, ndarray[np.uint8_t, ndim=2, cast=True] data, list group_locs):
    cdef int i, j, k = 0
    cdef int nr = data.shape[0]
    cdef int nc = data.shape[1]
    cdef int nc_final = nc - len(group_locs)
    cdef ndarray[np.uint8_t, ndim=2, cast=True] result = np.empty((nr, nc_final), dtype='bool')
    cdef ndarray[np.uint8_t, ndim=2, cast=True] cur_min = np.full((size, nc_final), True, dtype='bool')
    for i in range(nc):
        if i in group_locs:
            k += 1
            continue
        for j in range(nr):
            if cur_min[labels[j], i - k] == False:
                result[j, i - k] = False
                continue
            elif data[j, i] == False:
                cur_min[labels[j], i - k] = False
                result[j, i - k] = False
            else:
                result[j, i - k] = True
    return result

def cummin_date(ndarray[np.int64_t] labels, int size, ndarray[np.int64_t, ndim=2] data, list group_locs):
    cdef int i, j, k = 0
    cdef int nr = data.shape[0]
    cdef int nc = data.shape[1]
    cdef int nc_final = nc - len(group_locs)
    cdef ndarray[np.int64_t, ndim=2] result = np.empty((nr, nc_final), dtype='int64')
    cdef long nat = np.datetime64('nat').astype('int64')
    cdef ndarray[np.int64_t, ndim=2] cur_min = np.full((size, nc_final), nat, dtype='int64')
    for i in range(nc):
        if i in group_locs:
            k += 1
            continue
        for j in range(nr):
            if (data[j, i] < cur_min[labels[j], i - k] and data[j, i] != nat) or cur_min[labels[j], i - k] == nat:
                cur_min[labels[j], i - k] = data[j, i]
            result[j, i - k] = cur_min[labels[j], i - k]
    return result

def cumsum_int(ndarray[np.int64_t] labels, int size, ndarray[np.int64_t, ndim=2] data, list group_locs):
    cdef int i, j, k = 0
    cdef int nr = data.shape[0]
    cdef int nc = data.shape[1]
    cdef int nc_final = nc - len(group_locs)
    cdef ndarray[np.int64_t, ndim=2] result = np.zeros((nr, nc_final), dtype='int64')
    cdef ndarray[np.int64_t, ndim=2] cur_sum = np.zeros((size, nc_final), dtype='int64')
    for i in range(nc):
        if i in group_locs:
            k += 1
            continue
        for j in range(nr):
            cur_sum[labels[j], i - k] += data[j, i]
            result[j, i - k] = cur_sum[labels[j], i - k]
    return result

def cumsum_float(ndarray[np.int64_t] labels, int size, ndarray[np.float64_t, ndim=2] data, list group_locs):
    cdef int i, j, k = 0
    cdef int nr = data.shape[0]
    cdef int nc = data.shape[1]
    cdef int nc_final = nc - len(group_locs)
    cdef ndarray[np.float64_t, ndim=2] result = np.zeros((nr, nc_final), dtype='float64')
    cdef ndarray[np.float64_t, ndim=2] cur_sum = np.zeros((size, nc_final), dtype='float64')
    for i in range(nc):
        if i in group_locs:
            k += 1
            continue
        for j in range(nr):
            if not isnan(data[j, i]):
                cur_sum[labels[j], i - k] += data[j, i]
            result[j, i - k] = cur_sum[labels[j], i - k]
    return result

def cumsum_bool(ndarray[np.int64_t] labels, int size, ndarray[np.uint8_t, ndim=2, cast=True] data, list group_locs):
    cdef int i, j, k = 0
    cdef int nr = data.shape[0]
    cdef int nc = data.shape[1]
    cdef int nc_final = nc - len(group_locs)
    cdef ndarray[np.int64_t, ndim=2] result = np.zeros((nr, nc_final), dtype='int64')
    cdef ndarray[np.int64_t, ndim=2] cur_sum = np.zeros((size, nc_final), dtype='int64')
    for i in range(nc):
        if i in group_locs:
            k += 1
            continue
        for j in range(nr):
                cur_sum[labels[j], i - k] += data[j, i]
                result[j, i - k] = cur_sum[labels[j], i - k]
    return result



def cumprod_int(ndarray[np.int64_t] labels, int size, ndarray[np.int64_t, ndim=2] data, list group_locs):
    cdef int i, j, k = 0
    cdef int nr = data.shape[0]
    cdef int nc = data.shape[1]
    cdef int nc_final = nc - len(group_locs)
    cdef ndarray[np.int64_t, ndim=2] result = np.ones((nr, nc_final), dtype='int64')
    cdef ndarray[np.int64_t, ndim=2] cur_sum = np.ones((size, nc_final), dtype='int64')
    for i in range(nc):
        if i in group_locs:
            k += 1
            continue
        for j in range(nr):
            cur_sum[labels[j], i - k] *= data[j, i]
            result[j, i - k] = cur_sum[labels[j], i - k]
    return result

def cumprod_float(ndarray[np.int64_t] labels, int size, ndarray[np.float64_t, ndim=2] data, list group_locs):
    cdef int i, j, k = 0
    cdef int nr = data.shape[0]
    cdef int nc = data.shape[1]
    cdef int nc_final = nc - len(group_locs)
    cdef ndarray[np.float64_t, ndim=2] result = np.ones((nr, nc_final), dtype='float64')
    cdef ndarray[np.float64_t, ndim=2] cur_sum = np.ones((size, nc_final), dtype='float64')
    for i in range(nc):
        if i in group_locs:
            k += 1
            continue
        for j in range(nr):
            if not isnan(data[j, i]):
                cur_sum[labels[j], i - k] *= data[j, i]
            result[j, i - k] = cur_sum[labels[j], i - k]
    return result

def cumprod_bool(ndarray[np.int64_t] labels, int size, ndarray[np.uint8_t, ndim=2, cast=True] data, list group_locs):
    cdef int i, j, k = 0
    cdef int nr = data.shape[0]
    cdef int nc = data.shape[1]
    cdef int nc_final = nc - len(group_locs)
    cdef ndarray[np.int64_t, ndim=2] result = np.ones((nr, nc_final), dtype='int64')
    cdef ndarray[np.int64_t, ndim=2] cur_sum = np.ones((size, nc_final), dtype='int64')
    for i in range(nc):
        if i in group_locs:
            k += 1
            continue
        for j in range(nr):
                cur_sum[labels[j], i - k] *= data[j, i]
                result[j, i - k] = cur_sum[labels[j], i - k]
    return result


def custom_int(ndarray[np.int64_t] labels, int size, ndarray[np.int64_t, ndim=2] data,
               list group_locs, func, col_dict):
    cdef int i, j, k = 0
    cdef long start=0, end, xlen, med_idx
    cdef np.float64_t first, second
    cdef int nr = data.shape[0]
    cdef int nc = data.shape[1]
    cdef ndarray[np.int64_t] label_count = np.zeros(size, dtype='int64')
    cdef ndarray[np.int64_t] label_cumsum = np.empty(size, dtype='int64')
    cdef ndarray[np.int64_t] label_locs = np.empty(nr, dtype='int64')
    cdef ndarray[np.int64_t] label_count_cur = np.zeros(size, dtype='int64')
    cdef ndarray[np.int64_t] x
    cdef ndarray result
    cdef bint is_first_group = True

    from dexplo._frame import DataFrame

    for i in range(nr):
        label_count[labels[i]] += 1

    label_cumsum = np.roll(label_count, 1)
    label_cumsum[0] = 0
    label_cumsum = label_cumsum.cumsum()

    for i in range(nr):
        label_locs[label_cumsum[labels[i]] + label_count_cur[labels[i]]] = i
        label_count_cur[labels[i]] += 1

    for i in range(size):
        end = start + label_count[i]
        k = 0
        for j in range(nc):
            if j in group_locs:
                k += 1
                continue
            x = data[:, j][label_locs[start:end]]

            new_data = {'i': x[:, np.newaxis]}
            col_name = col_dict[j]
            col_info = {col_name: _utils.Column('i', 0, 0)}
            df = DataFrame._construct_from_new(new_data, col_info, [col_name])

            if is_first_group:
                first_result = func(df)

                if isinstance(first_result, (DataFrame, ndarray)):
                    if first_result.size != 1:
                        raise ValueError('When calling `agg` with a custom function, you must '
                                         'return a scalar value')

                    if isinstance(first_result, DataFrame):
                        first_result = first_result[0, 0]
                    else:
                        first_result = first_result.flat[0]

                if isinstance(first_result, (bool, np.bool_)):
                    result = np.empty((size, nc - len(group_locs)), dtype='bool')
                    dtype = 'b'
                elif isinstance(first_result, (np.integer, int)):
                    result = np.empty((size, nc - len(group_locs)), dtype='int64')
                    dtype = 'i'
                elif isinstance(first_result, (np.floating, float, np.number)):
                    result = np.empty((size, nc - len(group_locs)), dtype='float64')
                    dtype = 'f'
                elif isinstance(first_result, (str, type(None))):
                    result = np.empty((size, nc - len(group_locs)), dtype='O')
                    dtype = 'O'
                else:
                    raise TypeError(f'You returned the datatype {type(first_result)} from the '
                                    '`agg` method which is unable '
                                    'to be placed inside a DataFrame. Please return either a '
                                    'one element DataFrame/ndarray or an int, float, '
                                    'boolean, string, None, datetime, or timedelta. ')

                is_first_group = False
                result[i, j - k] = first_result
            else:
                next_result = func(df)
                if isinstance(next_result, (DataFrame, ndarray)):
                    if next_result.size != 1:
                        raise ValueError('When calling `agg` with a custom function, you must '
                                         'return a scalar value')

                    if isinstance(next_result, DataFrame):
                        next_result = next_result[0, 0]
                    else:
                        next_result = next_result.flat[0]

                if isinstance(next_result, (bool, np.bool_)):
                    if dtype == 'O':
                        raise TypeError('When aggregating, the return value for each column '
                                        'must be of the same type. You have a string mixed '
                                        'with a boolean.')
                elif isinstance(next_result, (np.integer, int)):
                    if dtype == 'b':
                        result = result.astype('int64')
                        dtype = 'i'
                    elif dtype == 'O':
                        raise TypeError('When aggregating, the return value for each column '
                                        'must be of the same type. You have a string mixed '
                                        'with an integer.')
                elif isinstance(next_result, (np.floating, float, np.number)):
                    if dtype == 'b' or dtype == 'i':
                        result = result.astype('float64')
                        dtype = 'f'
                    elif dtype == 'O':
                        raise TypeError('When aggregating, the return value for each column '
                                        'must be of the same type. You have a string mixed '
                                        'with a float.')
                elif isinstance(next_result, (str, type(None))):
                    if dtype != 'O':
                        raise TypeError('When aggregating, the return value for each column '
                                        'must be of the same type. You have a string mixed '
                                        'with non-strings')
                else:
                    raise TypeError(f'You returned the datatype {type(next_result)} from the '
                                    '`agg` method which is unable '
                                    'to be placed inside a DataFrame. Please return either a '
                                    'one element DataFrame/ndarray or an int, float, '
                                    'boolean, string, None, datetime, or timedelta. ')

                result[i, j - k] = next_result

        start = end
    return result

def custom_float(ndarray[np.int64_t] labels, int size, ndarray[np.float64_t, ndim=2] data,
                 list group_locs, func, col_dict):
    cdef int i, j, k = 0
    cdef long start=0, end, xlen, med_idx
    cdef np.float64_t first, second
    cdef int nr = data.shape[0]
    cdef int nc = data.shape[1]
    cdef ndarray[np.int64_t] label_count = np.zeros(size, dtype='int64')
    cdef ndarray[np.int64_t] label_cumsum = np.empty(size, dtype='int64')
    cdef ndarray[np.int64_t] label_locs = np.empty(nr, dtype='int64')
    cdef ndarray[np.int64_t] label_count_cur = np.zeros(size, dtype='int64')
    cdef ndarray[np.float64_t] x
    cdef ndarray result
    cdef bint is_first_group = True

    from dexplo._frame import DataFrame

    for i in range(nr):
        label_count[labels[i]] += 1

    label_cumsum = np.roll(label_count, 1)
    label_cumsum[0] = 0
    label_cumsum = label_cumsum.cumsum()

    for i in range(nr):
        label_locs[label_cumsum[labels[i]] + label_count_cur[labels[i]]] = i
        label_count_cur[labels[i]] += 1

    for i in range(size):
        end = start + label_count[i]
        k = 0
        for j in range(nc):
            if j in group_locs:
                k += 1
                continue

            x = data[:, j][label_locs[start:end]]
            new_data = {'f': x[:, np.newaxis]}
            col_name = col_dict[j]
            col_info = {col_name: _utils.Column('f', 0, 0)}

            df = DataFrame._construct_from_new(new_data, col_info, [col_name])

            if is_first_group:
                first_result = func(df)
                if isinstance(first_result, (DataFrame, ndarray)):
                    if first_result.size != 1:
                        raise ValueError('When calling `agg` with a custom function, you must '
                                         'return a scalar value')

                    if isinstance(first_result, DataFrame):
                        first_result = first_result[0, 0]
                    else:
                        first_result = first_result.flat[0]

                if isinstance(first_result, (bool, np.bool_)):
                    result = np.empty((size, nc - len(group_locs)), dtype='bool')
                    dtype = 'b'
                elif isinstance(first_result, (np.integer, int)):
                    result = np.empty((size, nc - len(group_locs)), dtype='int64')
                    dtype = 'i'
                elif isinstance(first_result, (np.floating, float, np.number)):
                    result = np.empty((size, nc - len(group_locs)), dtype='float64')
                    dtype = 'f'
                elif isinstance(first_result, (str, type(None))):
                    result = np.empty((size, nc - len(group_locs)), dtype='O')
                    dtype = 'O'
                else:
                    raise TypeError(f'You returned the datatype {type(first_result)} which is unable'
                                    'to be placed inside a DataFrame. Please return either a'
                                    'one element DataFrame/ndarray or an int, float, '
                                    'boolean, string, None, datetime, or timedelta')

                is_first_group = False
                result[i, j - k] = first_result
            else:
                next_result = func(df)
                if isinstance(next_result, (DataFrame, ndarray)):
                    if next_result.size != 1:
                        raise ValueError('When calling `agg` with a custom function, you must '
                                         'return a scalar value')

                    if isinstance(next_result, DataFrame):
                        next_result = next_result[0, 0]
                    else:
                        next_result = next_result.flat[0]

                if isinstance(next_result, (bool, np.bool_)):
                    if dtype == 'O':
                        raise TypeError('When aggregating, the return value for each column '
                                        'must be of the same type. You have a string mixed '
                                        'with a boolean.')
                elif isinstance(next_result, (np.integer, int)):
                    if dtype == 'b':
                        result = result.astype('int64')
                        dtype = 'i'
                    elif dtype == 'O':
                        raise TypeError('When aggregating, the return value for each column '
                                        'must be of the same type. You have a string mixed '
                                        'with an integer.')
                elif isinstance(next_result, (np.floating, float, np.number)):
                    if dtype == 'b' or dtype == 'i':
                        result = result.astype('float64')
                        dtype = 'f'
                    elif dtype == 'O':
                        raise TypeError('When aggregating, the return value for each column '
                                        'must be of the same type. You have a string mixed '
                                        'with a float.')
                elif isinstance(next_result, (str, type(None))):
                    if dtype != 'O':
                        raise TypeError('When aggregating, the return value for each column '
                                        'must be of the same type. You have a string mixed '
                                        'with non-strings')
                else:
                    raise TypeError(f'You returned the datatype {type(next_result)} from the '
                                    '`agg` method which is unable '
                                    'to be placed inside a DataFrame. Please return either a '
                                    'one element DataFrame/ndarray or an int, float, '
                                    'boolean, string, None, datetime, or timedelta. ')

                result[i, j - k] = next_result

        start = end
    return result


def custom_str(ndarray[np.int64_t] labels, int size, ndarray[object, ndim=2] data,
                 list group_locs, func, col_dict):
    cdef int i, j, k = 0
    cdef long start=0, end, xlen, med_idx
    cdef np.float64_t first, second
    cdef int nr = data.shape[0]
    cdef int nc = data.shape[1]
    cdef ndarray[np.int64_t] label_count = np.zeros(size, dtype='int64')
    cdef ndarray[np.int64_t] label_cumsum = np.empty(size, dtype='int64')
    cdef ndarray[np.int64_t] label_locs = np.empty(nr, dtype='int64')
    cdef ndarray[np.int64_t] label_count_cur = np.zeros(size, dtype='int64')
    cdef ndarray[object] x
    cdef ndarray result
    cdef bint is_first_group = True

    from dexplo._frame import DataFrame

    for i in range(nr):
        label_count[labels[i]] += 1

    label_cumsum = np.roll(label_count, 1)
    label_cumsum[0] = 0
    label_cumsum = label_cumsum.cumsum()

    for i in range(nr):
        label_locs[label_cumsum[labels[i]] + label_count_cur[labels[i]]] = i
        label_count_cur[labels[i]] += 1

    for i in range(size):
        end = start + label_count[i]
        k = 0
        for j in range(nc):
            if j in group_locs:
                k += 1
                continue

            x = data[:, j][label_locs[start:end]]
            new_data = {'O': x[:, np.newaxis]}
            col_name = col_dict[j]
            col_info = {col_name: _utils.Column('O', 0, 0)}

            df = DataFrame._construct_from_new(new_data, col_info, [col_name])

            if is_first_group:
                first_result = func(df)
                if isinstance(first_result, (DataFrame, ndarray)):
                    if first_result.size != 1:
                        raise ValueError('When calling `agg` with a custom function, you must '
                                         'return a scalar value')

                    if isinstance(first_result, DataFrame):
                        first_result = first_result[0, 0]
                    else:
                        first_result = first_result.flat[0]

                if isinstance(first_result, (bool, np.bool_)):
                    result = np.empty((size, nc - len(group_locs)), dtype='bool')
                    dtype = 'b'
                elif isinstance(first_result, (np.integer, int)):
                    result = np.empty((size, nc - len(group_locs)), dtype='int64')
                    dtype = 'i'
                elif isinstance(first_result, (np.floating, float, np.number)):
                    result = np.empty((size, nc - len(group_locs)), dtype='float64')
                    dtype = 'f'
                elif isinstance(first_result, (str, type(None))):
                    result = np.empty((size, nc - len(group_locs)), dtype='O')
                    dtype = 'O'
                else:
                    raise TypeError(f'You returned the datatype {type(first_result)} from the '
                                    '`agg` method which is unable '
                                    'to be placed inside a DataFrame. Please return either a '
                                    'one element DataFrame/ndarray or an int, float, '
                                    'boolean, string, None, datetime, or timedelta. ')

                is_first_group = False
                result[i, j - k] = first_result
            else:
                next_result = func(df)
                if isinstance(next_result, (DataFrame, ndarray)):
                    if next_result.size != 1:
                        raise ValueError('When calling `agg` with a custom function, you must '
                                         'return a scalar value')

                    if isinstance(next_result, DataFrame):
                        next_result = next_result[0, 0]
                    else:
                        next_result = next_result.flat[0]

                if isinstance(next_result, (bool, np.bool_)):
                    if dtype == 'O':
                        raise TypeError('When aggregating, the return value for each column '
                                        'must be of the same type. You have a string mixed '
                                        'with a boolean.')
                elif isinstance(next_result, (np.integer, int)):
                    if dtype == 'b':
                        result = result.astype('int64')
                        dtype = 'i'
                    elif dtype == 'O':
                        raise TypeError('When aggregating, the return value for each column '
                                        'must be of the same type. You have a string mixed '
                                        'with an integer.')
                elif isinstance(next_result, (np.floating, float, np.number)):
                    if dtype == 'b' or dtype == 'i':
                        result = result.astype('float64')
                        dtype = 'f'
                    elif dtype == 'O':
                        raise TypeError('When aggregating, the return value for each column '
                                        'must be of the same type. You have a string mixed '
                                        'with a float.')
                elif isinstance(next_result, (str, type(None))):
                    if dtype != 'O':
                        raise TypeError('When aggregating, the return value for each column '
                                        'must be of the same type. You have a string mixed '
                                        'with non-strings')
                else:
                    raise TypeError(f'You returned the datatype {type(next_result)} from the '
                                    '`agg` method which is unable '
                                    'to be placed inside a DataFrame. Please return either a '
                                    'one element DataFrame/ndarray or an int, float, '
                                    'boolean, string, None, datetime, or timedelta. ')

                result[i, j - k] = next_result

        start = end
    return result

def custom_bool(ndarray[np.int64_t] labels, int size, ndarray[np.uint8_t, ndim=2, cast=True] data,
                 list group_locs, func, col_dict):
    cdef int i, j, k = 0
    cdef long start=0, end, xlen, med_idx
    cdef np.float64_t first, second
    cdef int nr = data.shape[0]
    cdef int nc = data.shape[1]
    cdef ndarray[np.int64_t] label_count = np.zeros(size, dtype='int64')
    cdef ndarray[np.int64_t] label_cumsum = np.empty(size, dtype='int64')
    cdef ndarray[np.int64_t] label_locs = np.empty(nr, dtype='int64')
    cdef ndarray[np.int64_t] label_count_cur = np.zeros(size, dtype='int64')
    cdef ndarray[np.uint8_t, cast=True] x
    cdef ndarray result
    cdef bint is_first_group = True

    from dexplo._frame import DataFrame

    for i in range(nr):
        label_count[labels[i]] += 1

    label_cumsum = np.roll(label_count, 1)
    label_cumsum[0] = 0
    label_cumsum = label_cumsum.cumsum()

    for i in range(nr):
        label_locs[label_cumsum[labels[i]] + label_count_cur[labels[i]]] = i
        label_count_cur[labels[i]] += 1

    for i in range(size):
        end = start + label_count[i]
        k = 0
        for j in range(nc):
            if j in group_locs:
                k += 1
                continue

            x = data[:, j][label_locs[start:end]]
            new_data = {'b': x[:, np.newaxis]}
            col_name = col_dict[j]
            col_info = {col_name: _utils.Column('b', 0, 0)}

            df = DataFrame._construct_from_new(new_data, col_info, [col_name])

            if is_first_group:
                first_result = func(df)
                if isinstance(first_result, (DataFrame, ndarray)):
                    if first_result.size != 1:
                        raise ValueError('When calling `agg` with a custom function, you must '
                                         'return a scalar value')

                    if isinstance(first_result, DataFrame):
                        first_result = first_result[0, 0]
                    else:
                        first_result = first_result.flat[0]

                if isinstance(first_result, (bool, np.bool_)):
                    result = np.empty((size, nc - len(group_locs)), dtype='bool')
                    dtype = 'b'
                elif isinstance(first_result, (np.integer, int)):
                    result = np.empty((size, nc - len(group_locs)), dtype='int64')
                    dtype = 'i'
                elif isinstance(first_result, (np.floating, float, np.number)):
                    result = np.empty((size, nc - len(group_locs)), dtype='float64')
                    dtype = 'f'
                elif isinstance(first_result, (str, type(None))):
                    result = np.empty((size, nc - len(group_locs)), dtype='O')
                    dtype = 'O'
                else:
                    raise TypeError(f'You returned the datatype {type(first_result)} from the '
                                    '`agg` method which is unable '
                                    'to be placed inside a DataFrame. Please return either a '
                                    'one element DataFrame/ndarray or an int, float, '
                                    'boolean, string, None, datetime, or timedelta. ')


                is_first_group = False
                result[i, j - k] = first_result
            else:
                next_result = func(df)
                if isinstance(next_result, (DataFrame, ndarray)):
                    if next_result.size != 1:
                        raise ValueError('When calling `agg` with a custom function, you must '
                                         'return a scalar value')

                    if isinstance(next_result, DataFrame):
                        next_result = next_result[0, 0]
                    else:
                        next_result = next_result.flat[0]

                if isinstance(next_result, (bool, np.bool_)):
                    if dtype == 'O':
                        raise TypeError('When aggregating, the return value for each column '
                                        'must be of the same type. You have a string mixed '
                                        'with a boolean.')
                elif isinstance(next_result, (np.integer, int)):
                    if dtype == 'b':
                        result = result.astype('int64')
                        dtype = 'i'
                    elif dtype == 'O':
                        raise TypeError('When aggregating, the return value for each column '
                                        'must be of the same type. You have a string mixed '
                                        'with an integer.')
                elif isinstance(next_result, (np.floating, float, np.number)):
                    if dtype == 'b' or dtype == 'i':
                        result = result.astype('float64')
                        dtype = 'f'
                    elif dtype == 'O':
                        raise TypeError('When aggregating, the return value for each column '
                                        'must be of the same type. You have a string mixed '
                                        'with a float.')
                elif isinstance(next_result, (str, type(None))):
                    if dtype != 'O':
                        raise TypeError('When aggregating, the return value for each column '
                                        'must be of the same type. You have a string mixed '
                                        'with non-strings')
                else:
                    raise TypeError(f'You returned the datatype {type(next_result)} from the '
                                    '`agg` method which is unable '
                                    'to be placed inside a DataFrame. Please return either a '
                                    'one element DataFrame/ndarray or an int, float, '
                                    'boolean, string, None, datetime, or timedelta. ')

                result[i, j - k] = next_result

        start = end
    return result

def filter(ndarray[np.int64_t] labels, int size, df, func, *args, **kwargs):
    cdef int i, j
    cdef long start=0, end
    cdef int nr = df.shape[0]
    cdef int nc = df.shape[1]
    cdef ndarray[np.int64_t] label_count = np.zeros(size, dtype='int64')
    cdef ndarray[np.int64_t] label_cumsum = np.empty(size, dtype='int64')
    cdef ndarray[np.int64_t] label_locs = np.empty(nr, dtype='int64')
    cdef ndarray[np.int64_t] label_count_cur = np.zeros(size, dtype='int64')
    cdef ndarray[np.uint8_t, cast=True] result = np.zeros(nr, dtype='bool')

    from dexplo._frame import DataFrame

    for i in range(nr):
        label_count[labels[i]] += 1

    label_cumsum = np.roll(label_count, 1)
    label_cumsum[0] = 0
    label_cumsum = label_cumsum.cumsum()

    for i in range(nr):
        label_locs[label_cumsum[labels[i]] + label_count_cur[labels[i]]] = i
        label_count_cur[labels[i]] += 1

    for i in range(size):
        end = start + label_count[i]

        locs = label_locs[start:end]
        new_data = {kind: data[locs] for kind, data in df._data.items()}
        col_info = df._column_info
        columns = df._columns
        cur_df = DataFrame._construct_from_new(new_data, col_info, columns)

        next_result = func(cur_df, *args, **kwargs)

        if isinstance(next_result, (DataFrame, ndarray)):
            if next_result.size != 1:
                raise TypeError('When calling `filter`, you must return a scalar boolean value')

            if isinstance(next_result, DataFrame):
                next_result = next_result[0, 0]
            else:
                next_result = next_result.flat[0]

        if not isinstance(next_result, (bool, np.bool_)):
            raise TypeError('When calling `filter`, you must return a scalar boolean value')

        if next_result:
            result[locs] = True

        start = end
    return result

def apply(ndarray[np.int64_t] labels, int size, df, func, *args, **kwargs):
    cdef int i, j
    cdef long start=0, end
    cdef int nr = df.shape[0]
    cdef int nc = df.shape[1]
    cdef ndarray[np.int64_t] label_count = np.zeros(size, dtype='int64')
    cdef ndarray[np.int64_t] label_cumsum = np.empty(size, dtype='int64')
    cdef ndarray[np.int64_t] label_locs = np.empty(nr, dtype='int64')
    cdef ndarray[np.int64_t] label_count_cur = np.zeros(size, dtype='int64')
    cdef ndarray[np.uint8_t, cast=True] result = np.zeros(nr, dtype='bool')
    cdef int result_nc
    cdef ndarray result_columns
    cdef bint returns_df = False
    cdef bint all_dfs_same = True
    cdef list return_items = []
    cdef list column_dtype_final
    cdef ndarray[np.int64_t] group_repeats = np.zeros(size, dtype='int64')

    from dexplo._frame import DataFrame

    for i in range(nr):
        label_count[labels[i]] += 1

    label_cumsum = np.roll(label_count, 1)
    label_cumsum[0] = 0
    label_cumsum = label_cumsum.cumsum()

    for i in range(nr):
        label_locs[label_cumsum[labels[i]] + label_count_cur[labels[i]]] = i
        label_count_cur[labels[i]] += 1

    for i in range(size):
        end = start + label_count[i]
        locs = label_locs[start:end]
        new_data = {kind: data[locs] for kind, data in df._data.items()}
        col_info = df._column_info
        columns = df._columns
        cur_df = DataFrame._construct_from_new(new_data, col_info, columns)

        if i == 0:
            next_result = func(cur_df, *args, **kwargs)
            if isinstance(next_result, DataFrame):
                group_repeats[i] = next_result.shape[0]
                result_nc = next_result.shape[1]
                result_columns = next_result._columns
                returns_df = True
                column_dtype_final = [(col, next_result._column_info[col].dtype)
                                      for col in next_result._columns]
            elif isinstance(next_result, ndarray):
                if result.ndim == 2:
                    result_nc = next_result.shape[1]
                elif result.ndim == 1:
                    next_result = next_result[:, np.newaxis]
                    result_nc = 1
                else:
                    raise ValueError('You returned an array with more than 2 dimensions')
                group_repeats[i] = next_result.shape[0]
                dtype = next_result.dtype.kind
            elif isinstance(next_result, (int, np.integer)):
                group_repeats[i] = 1
                result_nc = 1
                next_result = np.array([[next_result]])
                dtype = 'i'

            elif isinstance(next_result, (bool, np.bool_)):
                group_repeats[i] = 1
                result_nc = 1
                next_result = np.array([[next_result]])
                dtype = 'b'
            elif isinstance(next_result, (float, np.floating)):
                group_repeats[i] = 1
                result_nc = 1
                next_result = np.array([[next_result]])
                dtype = 'f'
            elif isinstance(next_result, (str, type(None))):
                group_repeats[i] = 1
                result_nc = 1
                next_result = np.array([[next_result]])
                dtype = 'O'
            else:
                raise TypeError(f'The result from your apply function is {type(next_result)} '
                                'which is not supported. Return a DataFrane, NumPy array or '
                                'scalar value')
            return_items.append(next_result)
        else:
            next_result = func(cur_df, *args, **kwargs)
            if returns_df:
                if not isinstance(next_result, DataFrame):
                    raise TypeError('The first value returned from your apply function was a '
                                    'DataFrame. All subsequent objects returned must also '
                                    f'be DataFrames. This group returned a {type(next_result)}')
                if next_result.shape[1] != result_nc:
                    raise ValueError('The first DataFrame returned from your apply function '
                                     f'had {result_nc} columns, while the current group had '
                                     f'{next_result.shape[1]}. They must be equivalent')

                group_repeats[i] = next_result.shape[0]
                # compare the _column_info of next_result to first one
                first_column_info = return_items[0]._column_info
                cur_column_info = next_result._column_info
                if first_column_info != cur_column_info:
                    all_dfs_same = False
                    for i, col in enumerate(next_result._columns):
                        cur_dtype = next_result._column_info[col].dtype
                        final_dtype = column_dtype_final[i][1]
                        if final_dtype == 'O':
                            continue
                        if cur_dtype == 'b':
                            continue
                        if cur_dtype == 'i':
                            if final_dtype == 'b':
                                final_dtype == 'i'
                        if cur_dtype == 'f':
                            if final_dtype in 'ib':
                                final_dtype = 'f'

            elif isinstance(next_result, ndarray):
                if result.ndim == 2:
                    if next_result.shape[1] != result_nc:
                        raise ValueError('Your first returned array from the `apply` groupby '
                                         f'method had {result_nc} columns. Your current returned '
                                         f'array has {next_result.shape[1]} columns')
                elif result.ndim == 1:
                    if result_nc != 1:
                        raise ValueError('Your first returned array from the `apply` groupby '
                                         f'method had {result_nc} columns. Your current returned '
                                         f'array has 1 column')
                    next_result = next_result[:, np.newaxis]
                else:
                    raise ValueError('You returned an array with more than 2 dimensions')
                group_repeats[i] = next_result.shape[0]
            elif isinstance(next_result, (int, np.integer)):
                if result_nc != 1:
                        raise ValueError('Your first returned array from the `apply` groupby '
                                         f'method had {result_nc} columns. Your current returned '
                                         f'array has 1 column')
                next_result = np.array([[next_result]])
                group_repeats[i] = 1
            elif isinstance(next_result, (bool, np.bool_)):
                if result_nc != 1:
                        raise ValueError('Your first returned array from the `apply` groupby '
                                         f'method had {result_nc} columns. Your current returned '
                                         f'array has 1 column')
                next_result = np.array([[next_result]])
                group_repeats[i] = 1
            elif isinstance(next_result, (float, np.floating)):
                if result_nc != 1:
                        raise ValueError('Your first returned array from the `apply` groupby '
                                         f'method had {result_nc} columns. Your current returned '
                                         f'array has 1 column')
                next_result = np.array([[next_result]])
                group_repeats[i] = 1
            elif isinstance(next_result, (str, type(None))):
                if result_nc != 1:
                        raise ValueError('Your first returned array from the `apply` groupby '
                                         f'method had {result_nc} columns. Your current returned '
                                         f'array has 1 column')
                next_result = np.array([[next_result]])
                group_repeats[i] = 1
            else:
                raise TypeError(f'The result from your apply function is {type(next_result)} '
                                'which is not supported. Return a DataFrane, NumPy array or '
                                'scalar value')

            return_items.append(next_result)
        start = end
    if returns_df:
        if all_dfs_same:
            data_dict = defaultdict(list)
            for df in return_items:
                for dtype, data in df._data.items():
                    data_dict[dtype].append(data)
            new_data = {dtype: np.concatenate(data) for dtype, data in data_dict.items()}
            new_column_info = df._copy_column_info()
            new_columns = df._columns.copy()
    else:
        # we have a list of arrays
        for item in return_items:
            data = np.concatenate(return_items)
        if data.ndim == 1:
            data = data[:, np.newaxis]
        dtype = data.dtype.kind
        new_data = {dtype: data}
        new_columns = []
        for i in range(result_nc):
            new_columns.append('a' + str(i))

        new_column_info = {}
        for i in range(result_nc):
            new_column_info[new_columns[i]] = _utils.Column(dtype, i, i)

    return new_data, new_column_info, new_columns, group_repeats