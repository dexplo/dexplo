# adf
#cython: boundscheck=False
#cython: wraparound=False
import numpy as np
cimport numpy as np
from numpy cimport ndarray
from numpy import nan
import cython
from cpython cimport dict, set, list, tuple
from libc.math cimport isnan
import cmath
import groupby as gb
import math as _math
from .math import min_max_int, min_max_int2, isna_str, get_first_non_nan
from cpython.bytes cimport PyBytes_FromStringAndSize

try:
    import bottleneck as bn
except ImportError:
    import numpy as bn


def unique_str(ndarray[object] a):
    cdef int i, len_before
    cdef int n = len(a)
    cdef set s = set()
    cdef ndarray[np.uint8_t, cast = True] idx = np.zeros(n, dtype='bool')

    for i in range(n):
        len_before = len(s)
        s.add(a[i])
        if len(s) > len_before:
            idx[i] = True
    return idx

def unique_int(ndarray[np.int64_t] a):
    cdef int i, len_before
    cdef int n = len(a)
    cdef set s = set()
    cdef ndarray[np.uint8_t, cast = True] idx = np.zeros(n, dtype='bool')

    low, high = min_max_int(a)
    if high - low < 10_000_000:
        return unique_int_bounded(a, low, high)

    for i in range(n):
        len_before = len(s)
        s.add(a[i])
        if len(s) > len_before:
            idx[i] = True
    return idx

def unique_bool(ndarray[np.uint8_t, cast=True] a):
    cdef int i, n = len(a)
    cdef np.uint8_t first = a[0]
    cdef ndarray[np.uint8_t, cast = True] idx = np.zeros(n, dtype='bool')
    idx[0] = True
    for i in range(1, n):
        if a[i] != first:
            idx[i] = True
            break
    return idx

def unique_float(ndarray[double] a):
    cdef int i, len_before
    cdef int n = len(a)
    cdef set s = set()
    cdef ndarray[np.uint8_t, cast = True] idx = np.zeros(n, dtype='bool')

    for i in range(n):
        len_before = len(s)
        if isnan(a[i]):
            s.add(None)
        else:
            s.add(a[i])
        if len(s) > len_before:
            idx[i] = True
    return idx

def unique_int_bounded(ndarray[np.int64_t] a, np.int64_t low, np.int64_t high):
    cdef int i
    cdef int n = len(a)
    cdef ndarray[np.uint8_t, cast=True] unique
    cdef np.int64_t rng
    cdef ndarray[np.uint8_t, cast = True] idx = np.zeros(n, dtype='bool')

    rng = high - low + 1
    unique = np.zeros(rng, dtype='bool')

    for i in range(n):
        if not unique[a[i] - low]:
            unique[a[i] - low] = True
            idx[i] = True
    return idx

def unique_str_2d(ndarray[object, ndim=2] a):
    cdef int i, j, len_before
    cdef int nr = a.shape[0]
    cdef int nc = a.shape[1]
    cdef set s = set()
    cdef ndarray[np.uint8_t, cast = True] idx = np.zeros(nr, dtype='bool')
    cdef list val = list(range(nc))

    for i in range(nr):
        len_before = len(s)
        for j in range(nc):
            val[j] = a[i, j]
        s.add(tuple(val))
        if len(s) > len_before:
            idx[i] = True
    return idx

def unique_int_2d(ndarray[np.int64_t, ndim=2] a):
    cdef int i, len_before
    cdef int nr = a.shape[0]
    cdef int nc = a.shape[1]
    cdef set s = set()
    cdef ndarray[np.uint8_t, cast = True] idx = np.zeros(nr, dtype='bool')
    cdef long total_range = 1
    cdef long cur_range = 10 ** 7
    cdef int size = sizeof(np.int64_t) * nc

    lows, highs = min_max_int2(a, 0)

    ranges = highs - lows + 1
    for i in range(nc):
        cur_range /= ranges[i]
        total_range *= ranges[i]

    if cur_range > 1:
        return unique_int_bounded_2d(a, lows, highs, ranges, total_range)

    for i in range(nr):
        len_before = len(s)
        s.add(PyBytes_FromStringAndSize(<char*>&a[i, 0], size))
        if len(s) > len_before:
            idx[i] = True
    return idx

def unique_bool_2d(ndarray[np.uint8_t, cast=True, ndim=2] a):
    cdef int i, j
    cdef nr = a.shape[0]
    cdef nc = a.shape[1]
    cdef np.uint8_t first
    cdef ndarray[np.uint8_t, cast = True] idx = np.zeros(nr, dtype='bool')
    cdef ndarray[np.uint8_t, cast = True] unique
    cdef ndarray[np.int64_t] powers
    cdef long cur_total
    cdef set s = set()

    if nc <= 20:
        unique = np.zeros(2 ** nc, dtype='bool')
        powers = 2 ** np.arange(nc)
        for i in range(nr):
            cur_total = 0
            for j in range(nc):
                cur_total += powers[j] * a[i, j]
            if not unique[cur_total]:
                unique[cur_total] = True
                idx[i] = True
    else:
        for i in range(nr):
            len_before = len(s)
            s.add(tuple([a[i, j] for j in range(nc)]))
            if len(s) > len_before:
                idx[i] = True
    return idx

def unique_float_2d(ndarray[double, ndim=2] a):
    cdef int i, j, len_before, count = 0
    cdef int nr = a.shape[0]
    cdef int nc = a.shape[1]
    cdef set s = set()
    cdef ndarray[np.uint8_t, cast = True] idx = np.zeros(nr, dtype='bool')

    for i in range(nr):
        len_before = len(s)
        s.add(PyBytes_FromStringAndSize(<char*>&a[i, 0], sizeof(np.float64_t) * nc))
        if len(s) > len_before:
            idx[i] = True
    return idx

def unique_int_bounded_2d(ndarray[np.int64_t, ndim=2] a, ndarray[np.int64_t] lows,
                          ndarray[np.int64_t] highs, ndarray[np.int64_t] ranges, int total_range):
    cdef int i, j
    cdef int nr = a.shape[0]
    cdef int nc = a.shape[1]

    cdef ndarray[np.int64_t, ndim=2] idx
    cdef ndarray[np.uint8_t, cast=True] unique = np.zeros(total_range, dtype='bool')
    cdef ndarray[np.uint8_t, cast=True] pos = np.zeros(nr, dtype='bool')
    cdef long iloc
    cdef ndarray[np.int64_t] range_prod

    idx = a - lows
    range_prod = np.cumprod(ranges[:nc - 1])

    for i in range(nr):
        iloc = idx[i, 0]
        for j in range(nc - 1):
            iloc += range_prod[j] * idx[i, j + 1]
        if not unique[iloc]:
            # first time a group appears
            unique[iloc] = True
            pos[i] = True
    return pos

def unique_all_sep(ndarray[object, ndim = 2] a, ndarray[np.int64_t, ndim = 2] b,
                   ndarray[np.uint8_t, ndim = 2, cast = True] c, ndarray[np.float64_t, ndim = 2] d,
                   ndarray[np.int64_t] a_loc, ndarray[np.int64_t] b_loc, ndarray[np.int64_t] c_loc,
                   ndarray[np.int64_t] d_loc):
    cdef int i, j, len_before
    cdef int nr = a.shape[0]
    cdef int nca = len(a_loc)
    cdef int ncb = len(b_loc)
    cdef int ncc = len(c_loc)
    cdef int ncd = len(d_loc)
    cdef set s = set()
    cdef list v = list(range(nca + ncb + ncc + ncd))
    cdef ndarray[np.uint8_t, cast = True] idx = np.zeros(nr, dtype='bool')

    for i in range(nr):
        len_before = len(s)

        for j in range(nca):
            v[j] = a[i, a_loc[j]]
        for j in range(ncb):
            v[j + nca] = b[i, b_loc[j]]
        for j in range(ncc):
            v[j + nca + ncb] = c[i, c_loc[j]]
        for j in range(ncd):
            if isnan(d[i, d_loc[j]]):
                v[j + nca + ncb + ncc] = None
            else:
                v[j + nca + ncb + ncc] = d[i, d_loc[j]]

        s.add(tuple(v))
        if len(s) > len_before:
            idx[i] = True
    return idx

def unique_all_none(ndarray[object, ndim = 2] a, ndarray[np.int64_t, ndim = 2] b,
                   ndarray[np.uint8_t, ndim = 2, cast = True] c, ndarray[np.float64_t, ndim = 2] d,
                   ndarray[np.int64_t] a_loc, ndarray[np.int64_t] b_loc, ndarray[np.int64_t] c_loc,
                   ndarray[np.int64_t] d_loc):
    cdef int i, j, len_before, count = 0
    cdef int nr = a.shape[0]
    cdef int nca = len(a_loc)
    cdef int ncb = len(b_loc)
    cdef int ncc = len(c_loc)
    cdef int ncd = len(d_loc)
    cdef set s = set()
    cdef tuple t
    cdef dict d1 = {}
    cdef list v = list(range(nca + ncb + ncc + ncd))
    cdef ndarray[np.int64_t] group = np.empty(nr, dtype=np.int64)
    cdef ndarray[np.int64_t] counts = np.zeros(nr, dtype=np.int64)
    cdef ndarray[np.uint8_t, cast=True] keep = np.empty(nr, dtype='bool')

    for i in range(nr):
        len_before = len(s)

        for j in range(nca):
            v[j] = a[i, a_loc[j]]
        for j in range(ncb):
            v[j + nca] = b[i, b_loc[j]]
        for j in range(ncc):
            v[j + nca + ncb] = c[i, c_loc[j]]
        for j in range(ncd):
            if isnan(d[i, d_loc[j]]):
                v[j + nca + ncb + ncc] = None
            else:
                v[j + nca + ncb + ncc] = d[i, d_loc[j]]

        t = tuple(v)
        group[i] = d1.get(t, -1)
        if group[i] == -1:
            group[i] = count
            d1[t] = count
            count += 1
        counts[group[i]] += 1

    for i in range(nr):
        keep[i] = counts[group[i]] == 1

    return keep


def unique_float_string(ndarray[np.float64_t, ndim=2] f, ndarray[object, ndim=2] o):
    cdef int i, j, len_before
    cdef int nr = f.shape[0]
    cdef int ncf = f.shape[1]
    cdef int nco = o.shape[1]
    cdef set s = set()
    cdef list v = list(range(nco + 1))
    cdef ndarray[np.uint8_t, cast = True] idx = np.zeros(nr, dtype='bool')

    for i in range(nr):
        string = PyBytes_FromStringAndSize(<char*>&f[i, 0], sizeof(np.float64_t) * ncf)
        v[0] = string
        for j in range(nco):
            v[j + 1] = o[i, j]
        len_before = len(s)
        s.add(tuple(v))
        if len(s) > len_before:
            idx[i] = True
    return idx

def unique_int_string(ndarray[np.int64_t, ndim=2] a, ndarray[object, ndim=2] o):
    cdef int i, j, len_before
    cdef int nr = a.shape[0]
    cdef int nci = a.shape[1]
    cdef int nco = o.shape[1]
    cdef set s = set()
    cdef list v = list(range(nco + 1))
    cdef ndarray[np.uint8_t, cast = True] idx = np.zeros(nr, dtype='bool')

    for i in range(nr):
        string = PyBytes_FromStringAndSize(<char*>&a[i, 0], sizeof(np.int64_t) * nci)
        v[0] = string
        for j in range(nco):
            v[j + 1] = o[i, j]
        len_before = len(s)
        s.add(tuple(v))
        if len(s) > len_before:
            idx[i] = True
    return idx