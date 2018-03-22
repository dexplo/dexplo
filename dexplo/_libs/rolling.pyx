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


def sum_int(ndarray[np.int64_t, ndim=2] a, ndarray[np.int64_t] locs,
            int left, int right, int min_window):
    cdef:
        Py_ssize_t i=0, j, k, j_act
        Py_ssize_t nr = a.shape[0], nc = a.shape[1], nc_actual = len(locs)
        Py_ssize_t first_n=0, middle_n=nr, last_n=0
        ndarray[np.int64_t, ndim=2] result = np.zeros((nr, nc_actual), dtype='int64')
        np.int64_t total

    if left < 0:
        first_n = -left

    if right > 0:
        middle_n = nr - right
        last_n = right

    for j in range(nc_actual):
        j_act = locs[j]

        for i in range(first_n):
            total = 0
            for k in range(right + i):
                total += a[k, j_act]
            result[i, j] = total

        for i in range(first_n, middle_n):
            total = 0
            for k in range(i + left, i + right):
                total += a[k, j_act]
            result[i, j] = total

        for i in range(nr - last_n, nr):
            total = 0
            for k in range(i + left, nr):
                total += a[k, j_act]
            result[i, j] = total

    return result

def sum_float(ndarray[np.float64_t, ndim=2] a, ndarray[np.int64_t] locs,
            int left, int right, int min_window):
    cdef:
        Py_ssize_t i=0, j, j_act, k
        Py_ssize_t nr = a.shape[0], nc = a.shape[1], nc_actual = len(locs)
        Py_ssize_t first_n=0, middle_n=nr, last_n=0
        ndarray[np.float64_t, ndim=2] result = np.zeros((nr, nc_actual), dtype='float64')
        np.float64_t total

    if left < 0:
        first_n = -left

    if right > 0:
        middle_n = nr - right
        last_n = right

    for j in range(nc_actual):
        j_act = locs[j]

        for i in range(first_n):
            total = 0
            for k in range(right + i):
                if not npy_isnan(a[k, j_act]):
                    total += a[k, j_act]
            result[i, j] = total

        for i in range(first_n, middle_n):
            total = 0
            for k in range(i + left, i + right):
                if not npy_isnan(a[k, j_act]):
                    total += a[k, j_act]
            result[i, j] = total

        for i in range(nr - last_n, nr):
            total = 0
            for k in range(i + left, nr):
                if not npy_isnan(a[k, j_act]):
                    total += a[k, j_act]
            result[i, j] = total

    return result

def sum_bool(ndarray[np.uint8_t, ndim=2, cast=True] a, ndarray[np.int64_t] locs,
            int left, int right, int min_window):
    cdef:
        Py_ssize_t i=0, j, j_act, k
        Py_ssize_t nr = a.shape[0], nc = a.shape[1], nc_actual = len(locs)
        Py_ssize_t first_n=0, middle_n=nr, last_n=0
        ndarray[np.int64_t, ndim=2] result = np.zeros((nr, nc_actual), dtype='int64')
        np.int64_t total

    if left < 0:
        first_n = -left

    if right > 0:
        middle_n = nr - right
        last_n = right

    for j in range(nc_actual):
        j_act = locs[j]

        for i in range(first_n):
            total = 0
            for k in range(right + i):
                total += a[k, j_act]
            result[i, j] = total

        for i in range(first_n, middle_n):
            total = 0
            for k in range(i + left, i + right):
                total += a[k, j_act]
            result[i, j] = total

        for i in range(nr - last_n, nr):
            total = 0
            for k in range(i + left, nr):
                total += a[k, j_act]
            result[i, j] = total

    return result


def min_int(ndarray[np.int64_t, ndim=2] a, ndarray[np.int64_t] locs,
            int left, int right, int min_window):
    cdef:
        Py_ssize_t i=0, j, k, j_act
        Py_ssize_t nr = a.shape[0], nc = a.shape[1], nc_actual = len(locs)
        Py_ssize_t first_n=0, middle_n=nr, last_n=0
        np.int64_t amin

    if (left < 0 and right < 0) or (left > 0 and right > 0):
        result = np.full((nr, nc_actual), nan, dtype='float64')
    else:
        result = np.empty((nr, nc_actual), dtype='int64')

    if left < 0:
        first_n = -left

    if right > 0:
        middle_n = nr - right
        last_n = right

    for j in range(nc_actual):
        j_act = locs[j]

        for i in range(first_n):
            amin = MAX_INT
            for k in range(right + i):
                if a[k, j_act] < amin:
                    amin = a[k, j_act]
            result[i, j] = amin

        for i in range(first_n, middle_n):
            amin = MAX_INT
            for k in range(i + left, i + right):
                if a[k, j_act] < amin:
                    amin = a[k, j_act]
            result[i, j] = amin

        for i in range(nr - last_n, nr):
            amin = MAX_INT
            for k in range(i + left, nr):
                if a[k, j_act] < amin:
                    amin = a[k, j_act]
            result[i, j] = amin

    return result

def min_float(ndarray[np.float64_t, ndim=2] a, ndarray[np.int64_t] locs,
            int left, int right, int min_window):
    cdef:
        Py_ssize_t i=0, j, j_act, k
        Py_ssize_t nr = a.shape[0], nc = a.shape[1], nc_actual = len(locs)
        Py_ssize_t first_n=0, middle_n=nr, last_n=0
        ndarray[np.float64_t, ndim=2] result = np.full((nr, nc_actual), nan, dtype='float64')
        np.float64_t amin

    if left < 0:
        first_n = -left

    if right > 0:
        middle_n = nr - right
        last_n = right

    for j in range(nc_actual):
        j_act = locs[j]

        for i in range(first_n):
            amin = MAX_FLOAT
            for k in range(right + i):
                if a[k, j_act] < amin:
                    amin = a[k, j_act]
            if amin != MAX_FLOAT:
                result[i, j] = amin

        for i in range(first_n, middle_n):
            amin = MAX_FLOAT
            for k in range(i + left, i + right):
                if a[k, j_act] < amin:
                    amin = a[k, j_act]
            if amin != MAX_FLOAT:
                result[i, j] = amin

        for i in range(nr - last_n, nr):
            amin = MAX_FLOAT
            for k in range(i + left, nr):
                if a[k, j_act] < amin:
                    amin = a[k, j_act]
            if amin != MAX_FLOAT:
                result[i, j] = amin

    return result

def min_bool(ndarray[np.uint8_t, ndim=2, cast=True] a, ndarray[np.int64_t] locs,
            int left, int right, int min_window):
    cdef:
        Py_ssize_t i=0, j, j_act, k
        Py_ssize_t nr = a.shape[0], nc = a.shape[1], nc_actual = len(locs)
        Py_ssize_t first_n=0, middle_n=nr, last_n=0

    if (left < 0 and right < 0) or (left > 0 and right > 0):
        result = np.full((nr, nc_actual), nan, dtype='float64')
    else:
        result = np.ones((nr, nc_actual), dtype='bool')

    if left < 0:
        first_n = -left

    if right > 0:
        middle_n = nr - right
        last_n = right

    for j in range(nc_actual):
        j_act = locs[j]

        for i in range(first_n):
            for k in range(right + i):
                if not a[k, j_act]:
                    result[i, j] = False
                    break


        for i in range(first_n, middle_n):
            for k in range(i + left, i + right):
                if not a[k, j_act]:
                    result[i, j] = False
                    break

        for i in range(nr - last_n, nr):
            for k in range(i + left, nr):
                if not a[k, j_act]:
                    result[i, j] = False
                    break

    return result

def max_int(ndarray[np.int64_t, ndim=2] a, ndarray[np.int64_t] locs,
            int left, int right, int min_window):
    cdef:
        Py_ssize_t i=0, j, k, j_act
        Py_ssize_t nr = a.shape[0], nc = a.shape[1], nc_actual = len(locs)
        Py_ssize_t first_n=0, middle_n=nr, last_n=0
        np.int64_t amax

    if (left < 0 and right < 0) or (left > 0 and right > 0):
        result = np.full((nr, nc_actual), nan, dtype='float64')
    else:
        result = np.empty((nr, nc_actual), dtype='int64')

    if left < 0:
        first_n = -left

    if right > 0:
        middle_n = nr - right
        last_n = right

    for j in range(nc_actual):
        j_act = locs[j]

        for i in range(first_n):
            amax = MIN_INT
            for k in range(right + i):
                if a[k, j_act] > amax:
                    amax = a[k, j_act]
            result[i, j] = amax

        for i in range(first_n, middle_n):
            amax = MIN_INT
            for k in range(i + left, i + right):
                if a[k, j_act] > amax:
                    amax = a[k, j_act]
            result[i, j] = amax

        for i in range(nr - last_n, nr):
            amax = MIN_INT
            for k in range(i + left, nr):
                if a[k, j_act] > amax:
                    amax = a[k, j_act]
            result[i, j] = amax

    return result

def max_float(ndarray[np.float64_t, ndim=2] a, ndarray[np.int64_t] locs,
            int left, int right, int min_window):
    cdef:
        Py_ssize_t i=0, j, j_act, k
        Py_ssize_t nr = a.shape[0], nc = a.shape[1], nc_actual = len(locs)
        Py_ssize_t first_n=0, middle_n=nr, last_n=0
        ndarray[np.float64_t, ndim=2] result = np.full((nr, nc_actual), nan, dtype='float64')
        np.float64_t amax

    if left < 0:
        first_n = -left

    if right > 0:
        middle_n = nr - right
        last_n = right

    for j in range(nc_actual):
        j_act = locs[j]

        for i in range(first_n):
            amax = MIN_FLOAT
            for k in range(right + i):
                if a[k, j_act] > amax:
                    amax = a[k, j_act]
            if amax != MIN_FLOAT:
                result[i, j] = amax

        for i in range(first_n, middle_n):
            amax = MIN_FLOAT
            for k in range(i + left, i + right):
                if a[k, j_act] > amax:
                    amax = a[k, j_act]
            if amax != MIN_FLOAT:
                result[i, j] = amax

        for i in range(nr - last_n, nr):
            amax = MIN_FLOAT
            for k in range(i + left, nr):
                if a[k, j_act] > amax:
                    amax = a[k, j_act]
            if amax != MIN_FLOAT:
                result[i, j] = amax
    return result

def max_bool(ndarray[np.uint8_t, ndim=2, cast=True] a, ndarray[np.int64_t] locs,
            int left, int right, int min_window):
    cdef:
        Py_ssize_t i=0, j, j_act, k
        Py_ssize_t nr = a.shape[0], nc = a.shape[1], nc_actual = len(locs)
        Py_ssize_t first_n=0, middle_n=nr, last_n=0

    if (left < 0 and right < 0) or (left > 0 and right > 0):
        result = np.full((nr, nc_actual), nan, dtype='float64')
    else:
        result = np.zeros((nr, nc_actual), dtype='bool')

    if left < 0:
        first_n = -left

    if right > 0:
        middle_n = nr - right
        last_n = right

    for j in range(nc_actual):
        j_act = locs[j]

        for i in range(first_n):
            for k in range(right + i):
                if a[k, j_act]:
                    result[i, j] = True
                    break

        for i in range(first_n, middle_n):
            for k in range(i + left, i + right):
                if a[k, j_act]:
                    result[i, j] = True
                    break

        for i in range(nr - last_n, nr):
            for k in range(i + left, nr):
                if a[k, j_act]:
                    result[i, j] = True
                    break

    return result

def mean_int(ndarray[np.int64_t, ndim=2] a, ndarray[np.int64_t] locs,
            int left, int right, int min_window):
    cdef:
        Py_ssize_t i=0, j, k, j_act
        Py_ssize_t nr = a.shape[0], nc = a.shape[1], nc_actual = len(locs)
        Py_ssize_t first_n=0, middle_n=nr, last_n=0
        ndarray[np.float64_t, ndim=2] result = np.full((nr, nc_actual), nan, dtype='float64', order='F')
        np.int64_t total, n = right - left, n1

    if left < 0:
        first_n = -left

    if right > 0:
        middle_n = nr - right
        last_n = right

    for j in range(nc_actual):
        j_act = locs[j]

        for i in range(first_n):
            total = 0
            n1 = 0
            for k in range(right + i):
                total += a[k, j_act]
                n1 += 1
            result[i, j] = total / n1

        for i in range(first_n, middle_n):
            total = 0
            for k in range(i + left, i + right):
                total += a[k, j_act]
            result[i, j] = total / n

        for i in range(nr - last_n, nr):
            total = 0
            n1 = 0
            for k in range(i + left, nr):
                total += a[k, j_act]
                n1 += 1
            result[i, j] = total / n1

    return result

def mean_float(ndarray[np.float64_t, ndim=2] a, ndarray[np.int64_t] locs,
            int left, int right, int min_window):
    cdef:
        Py_ssize_t i=0, j, j_act, k
        Py_ssize_t nr = a.shape[0], nc = a.shape[1], nc_actual = len(locs)
        Py_ssize_t first_n=0, middle_n=nr, last_n=0
        ndarray[np.float64_t, ndim=2] result = np.full((nr, nc_actual), nan, dtype='float64')
        np.float64_t total, n

    if left < 0:
        first_n = -left

    if right > 0:
        middle_n = nr - right
        last_n = right

    for j in range(nc_actual):
        j_act = locs[j]

        for i in range(first_n):
            total = 0
            n = 0
            for k in range(right + i):
                if not npy_isnan(a[k, j_act]):
                    total += a[k, j_act]
                    n += 1
            result[i, j] = total / n

        for i in range(first_n, middle_n):
            total = 0
            n = 0
            for k in range(i + left, i + right):
                if not npy_isnan(a[k, j_act]):
                    total += a[k, j_act]
                    n += 1
            result[i, j] = total / n

        for i in range(nr - last_n, nr):
            total = 0
            n = 0
            for k in range(i + left, nr):
                if not npy_isnan(a[k, j_act]):
                    total += a[k, j_act]
                    n += 1
            result[i, j] = total / n

    return result

def mean_bool(ndarray[np.uint8_t, ndim=2, cast=True] a, ndarray[np.int64_t] locs,
            int left, int right, int min_window):
    cdef:
        Py_ssize_t i=0, j, j_act, k
        Py_ssize_t nr = a.shape[0], nc = a.shape[1], nc_actual = len(locs)
        Py_ssize_t first_n=0, middle_n=nr, last_n=0
        ndarray[np.float64_t, ndim=2] result = np.full((nr, nc_actual), nan, dtype='float64')
        np.float64_t total, n = right - left, n1

    if left < 0:
        first_n = -left

    if right > 0:
        middle_n = nr - right
        last_n = right

    for j in range(nc_actual):
        j_act = locs[j]

        for i in range(first_n):
            total = 0
            n1 = 0
            for k in range(right + i):
                total += a[k, j_act]
                n1 += 1
            result[i, j] = total / n1

        for i in range(first_n, middle_n):
            total = 0
            for k in range(i + left, i + right):
                total += a[k, j_act]
            result[i, j] = total / n

        for i in range(nr - last_n, nr):
            total = 0
            n1 = 0
            for k in range(i + left, nr):
                total += a[k, j_act]
                n1 += 1
            result[i, j] = total / n1

    return result


def count_int(ndarray[np.int64_t, ndim=2] a, ndarray[np.int64_t] locs,
            int left, int right, int min_window):
    cdef:
        Py_ssize_t i=0, j, k, j_act
        Py_ssize_t nr = a.shape[0], nc = a.shape[1], nc_actual = len(locs)
        Py_ssize_t first_n=0, middle_n=nr, last_n=0
        ndarray[np.int64_t, ndim=2] result = np.zeros((nr, nc_actual), dtype='int64')
        np.int64_t n, n1 = right - left

    if left < 0:
        first_n = -left

    if right > 0:
        middle_n = nr - right
        last_n = right

    for j in range(nc_actual):
        j_act = locs[j]

        for i in range(first_n):
            n = 0
            for k in range(right + i):
                n += 1
            result[i, j] = n

        for i in range(first_n, middle_n):
            result[i, j] = n1

        for i in range(nr - last_n, nr):
            n = 0
            for k in range(i + left, nr):
                n += 1
            result[i, j] = n

    return result

def count_float(ndarray[np.float64_t, ndim=2] a, ndarray[np.int64_t] locs,
            int left, int right, int min_window):
    cdef:
        Py_ssize_t i=0, j, j_act, k
        Py_ssize_t nr = a.shape[0], nc = a.shape[1], nc_actual = len(locs)
        Py_ssize_t first_n=0, middle_n=nr, last_n=0
        ndarray[np.int64_t, ndim=2] result = np.zeros((nr, nc_actual), dtype='int64')
        np.int64_t n

    if left < 0:
        first_n = -left

    if right > 0:
        middle_n = nr - right
        last_n = right

    for j in range(nc_actual):
        j_act = locs[j]

        for i in range(first_n):
            n = 0
            for k in range(right + i):
                if not npy_isnan(a[k, j_act]):
                    n += 1
            result[i, j] = n

        for i in range(first_n, middle_n):
            n = 0
            for k in range(i + left, i + right):
                if not npy_isnan(a[k, j_act]):
                    n += 1
            result[i, j] = n

        for i in range(nr - last_n, nr):
            n = 0
            for k in range(i + left, nr):
                if not npy_isnan(a[k, j_act]):
                    n += 1
            result[i, j] = n

    return result

def count_bool(ndarray[np.uint8_t, ndim=2, cast=True] a, ndarray[np.int64_t] locs,
            int left, int right, int min_window):
    cdef:
        Py_ssize_t i=0, j, j_act, k
        Py_ssize_t nr = a.shape[0], nc = a.shape[1], nc_actual = len(locs)
        Py_ssize_t first_n=0, middle_n=nr, last_n=0
        ndarray[np.int64_t, ndim=2] result = np.zeros((nr, nc_actual), dtype='int64')
        np.int64_t n, n1 = right - left

    if left < 0:
        first_n = -left

    if right > 0:
        middle_n = nr - right
        last_n = right

    for j in range(nc_actual):
        j_act = locs[j]

        for i in range(first_n):
            n = 0
            for k in range(right + i):
                n += 1
            result[i, j] = n

        for i in range(first_n, middle_n):
            result[i, j] = n1

        for i in range(nr - last_n, nr):
            n = 0
            for k in range(i + left, nr):
                n += 1
            result[i, j] = n

    return result

def count_str(ndarray[object, ndim=2] a, ndarray[np.int64_t] locs,
            int left, int right, int min_window):
    cdef:
        Py_ssize_t i=0, j, j_act, k
        Py_ssize_t nr = a.shape[0], nc = a.shape[1], nc_actual = len(locs)
        Py_ssize_t first_n=0, middle_n=nr, last_n=0
        ndarray[np.int64_t, ndim=2] result = np.zeros((nr, nc_actual), dtype='int64')
        np.int64_t n

    if left < 0:
        first_n = -left

    if right > 0:
        middle_n = nr - right
        last_n = right

    for j in range(nc_actual):
        j_act = locs[j]

        for i in range(first_n):
            n = 0
            for k in range(right + i):
                if a[k, j_act] is not None:
                    n += 1
            result[i, j] = n

        for i in range(first_n, middle_n):
            n = 0
            for k in range(i + left, i + right):
                if a[k, j_act] is not None:
                    n += 1
            result[i, j] = n

        for i in range(nr - last_n, nr):
            n = 0
            for k in range(i + left, nr):
                if a[k, j_act] is not None:
                    n += 1
            result[i, j] = n

    return result

def prod_int(ndarray[np.int64_t, ndim=2] a, ndarray[np.int64_t] locs,
            int left, int right, int min_window):
    cdef:
        Py_ssize_t i=0, j, k, j_act
        Py_ssize_t nr = a.shape[0], nc = a.shape[1], nc_actual = len(locs)
        Py_ssize_t first_n=0, middle_n=nr, last_n=0
        ndarray[np.int64_t, ndim=2] result = np.ones((nr, nc_actual), dtype='int64')
        np.int64_t total

    if left < 0:
        first_n = -left

    if right > 0:
        middle_n = nr - right
        last_n = right

    for j in range(nc_actual):
        j_act = locs[j]

        for i in range(first_n):
            total = 1
            for k in range(right + i):
                total *= a[k, j_act]
            result[i, j] = total

        for i in range(first_n, middle_n):
            total = 1
            for k in range(i + left, i + right):
                total *= a[k, j_act]
            result[i, j] = total

        for i in range(nr - last_n, nr):
            total = 1
            for k in range(i + left, nr):
                total *= a[k, j_act]
            result[i, j] = total

    return result

def prod_float(ndarray[np.float64_t, ndim=2] a, ndarray[np.int64_t] locs,
            int left, int right, int min_window):
    cdef:
        Py_ssize_t i=0, j, j_act, k
        Py_ssize_t nr = a.shape[0], nc = a.shape[1], nc_actual = len(locs)
        Py_ssize_t first_n=0, middle_n=nr, last_n=0
        ndarray[np.float64_t, ndim=2] result = np.ones((nr, nc_actual), dtype='float64')
        np.float64_t total

    if left < 0:
        first_n = -left

    if right > 0:
        middle_n = nr - right
        last_n = right

    for j in range(nc_actual):
        j_act = locs[j]

        for i in range(first_n):
            total = 1
            for k in range(right + i):
                if not npy_isnan(a[k, j_act]):
                    total *= a[k, j_act]
            result[i, j] = total

        for i in range(first_n, middle_n):
            total = 1
            for k in range(i + left, i + right):
                if not npy_isnan(a[k, j_act]):
                    total *= a[k, j_act]
            result[i, j] = total

        for i in range(nr - last_n, nr):
            total = 1
            for k in range(i + left, nr):
                if not npy_isnan(a[k, j_act]):
                    total *= a[k, j_act]
            result[i, j] = total

    return result

def prod_bool(ndarray[np.uint8_t, ndim=2, cast=True] a, ndarray[np.int64_t] locs,
            int left, int right, int min_window):
    cdef:
        Py_ssize_t i=0, j, j_act, k
        Py_ssize_t nr = a.shape[0], nc = a.shape[1], nc_actual = len(locs)
        Py_ssize_t first_n=0, middle_n=nr, last_n=0
        ndarray[np.int64_t, ndim=2] result = np.ones((nr, nc_actual), dtype='int64')
        np.int64_t total

    if left < 0:
        first_n = -left

    if right > 0:
        middle_n = nr - right
        last_n = right

    for j in range(nc_actual):
        j_act = locs[j]

        for i in range(first_n):
            total = 1
            for k in range(right + i):
                total *= a[k, j_act]
            result[i, j] = total

        for i in range(first_n, middle_n):
            total = 1
            for k in range(i + left, i + right):
                total *= a[k, j_act]
            result[i, j] = total

        for i in range(nr - last_n, nr):
            total = 1
            for k in range(i + left, nr):
                total *= a[k, j_act]
            result[i, j] = total

    return result