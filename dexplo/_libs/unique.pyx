#cython: boundscheck=False
#cython: wraparound=False
import numpy as np
cimport numpy as np
from numpy cimport ndarray
from cpython cimport dict, set, list, tuple
from libc.math cimport isnan
from .math import min_max_int, min_max_int2
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

def unique_float_string(ndarray[np.float64_t] f, ndarray[object, ndim=2] o):
    cdef int i, j, len_before
    cdef int nr = f.shape[0]
    cdef int nco = o.shape[1]
    cdef set s = set()
    cdef list v = list(range(nco + 1))
    cdef ndarray[np.uint8_t, cast = True] idx = np.zeros(nr, dtype='bool')

    for i in range(nr):
        if isnan(f[i]):
            v[0] = None
        else:
            v[0] = f[i]
        for j in range(nco):
            v[j + 1] = o[i, j]
        len_before = len(s)
        s.add(tuple(v))
        if len(s) > len_before:
            idx[i] = True
    return idx

def unique_int_string(ndarray[np.int64_t] a, ndarray[object, ndim=2] o):
    cdef int i, j, len_before
    cdef int nr = a.shape[0]
    cdef int nco = o.shape[1]
    cdef set s = set()
    cdef list v = list(range(nco + 1))
    cdef ndarray[np.uint8_t, cast = True] idx = np.zeros(nr, dtype='bool')

    for i in range(nr):
        v[0] = a[i]
        for j in range(nco):
            v[j + 1] = o[i, j]
        len_before = len(s)
        s.add(tuple(v))
        if len(s) > len_before:
            idx[i] = True
    return idx

def unique_bool_string(ndarray[np.uint8_t, cast=True] a, ndarray[object, ndim=2] o):
    cdef int i, j, len_before
    cdef int nr = a.shape[0]
    cdef int ncb = a.shape[1]
    cdef int nco = o.shape[1]
    cdef set s = set()
    cdef list v = list(range(nco + 1))
    cdef ndarray[np.uint8_t, cast = True] idx = np.zeros(nr, dtype='bool')

    for i in range(nr):
        v[0] = a[i]
        for j in range(nco):
            v[j + 1] = o[i, j]
        len_before = len(s)
        s.add(tuple(v))
        if len(s) > len_before:
            idx[i] = True
    return idx


def unique_float_string_2d(ndarray[np.float64_t, ndim=2] f, ndarray[object, ndim=2] o):
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

def unique_int_string_2d(ndarray[np.int64_t, ndim=2] a, ndarray[object, ndim=2] o):
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

def unique_bool_string_2d(ndarray[np.uint8_t, ndim=2, cast=True] a, ndarray[object, ndim=2] o):
    cdef int i, j, len_before
    cdef int nr = a.shape[0]
    cdef int ncb = a.shape[1]
    cdef int nco = o.shape[1]
    cdef set s = set()
    cdef list v = list(range(nco + 1))
    cdef ndarray[np.uint8_t, cast = True] idx = np.zeros(nr, dtype='bool')

    for i in range(nr):
        string = PyBytes_FromStringAndSize(<char*>&a[i, 0], sizeof(np.uint8_t) * ncb)
        v[0] = string
        for j in range(nco):
            v[j + 1] = o[i, j]
        len_before = len(s)
        s.add(tuple(v))
        if len(s) > len_before:
            idx[i] = True
    return idx

def unique_int_none(ndarray[np.int64_t] a):
    cdef int i, count = 0
    cdef int n = len(a)
    cdef dict d = {}
    cdef ndarray[np.int64_t] counts = np.zeros(n, dtype=np.int64)
    cdef ndarray[np.int64_t] group = np.zeros(n, dtype=np.int64)
    cdef ndarray[np.uint8_t, cast=True] keep = np.empty(n, dtype='bool')

    low, high = min_max_int(a)
    if high - low < 10_000_000:
        return unique_int_bounded_none(a, low, high)

    for i in range(n):
        group[i] = d.get(a[i], -1)
        if group[i] == -1:
            group[i] = count
            d[a[i]] = count
            count += 1
        counts[group[i]] += 1

    for i in range(n):
        keep[i] = counts[group[i]] == 1

    return keep

def unique_int_bounded_none(ndarray[np.int64_t] a, np.int64_t low, np.int64_t high):
    cdef int i, count = 0
    cdef int n = len(a)

    cdef np.int64_t rng
    cdef ndarray[np.int64_t] group
    cdef ndarray[np.uint8_t, cast=True] keep = np.empty(n, dtype='bool')
    cdef ndarray[np.int64_t] counts = np.zeros(n, dtype=np.int64)

    rng = high - low + 1
    group = np.full(rng, -1, dtype='int64')

    for i in range(n):
        if group[a[i] - low] == -1:
            group[a[i] - low] = count
            count += 1
        counts[group[a[i] - low]] += 1

    for i in range(n):
        keep[i] = counts[group[a[i] - low]] == 1

    return keep

def unique_str_none(ndarray[object] a):
    cdef int i, count = 0
    cdef int n = len(a)
    cdef dict d = {}
    cdef ndarray[np.int64_t] group = np.zeros(n, dtype=np.int64)
    cdef ndarray[np.int64_t] counts = np.zeros(n, dtype=np.int64)
    cdef ndarray[np.uint8_t, cast=True] keep = np.empty(n, dtype='bool')

    for i in range(n):
        group[i] = d.get(a[i], -1)
        if group[i] == -1:
            group[i] = count
            d[a[i]] = count
            count += 1
        counts[group[i]] += 1

    for i in range(n):
        keep[i] = counts[group[i]] == 1

    return keep


def unique_float_none(ndarray[np.float64_t] a):
    cdef int i, count = 0
    cdef int n = len(a)
    cdef dict d = {}
    cdef ndarray[np.int64_t] group = np.zeros(n, dtype=np.int64)
    cdef ndarray[np.int64_t] counts = np.zeros(n, dtype=np.int64)
    cdef ndarray[np.uint8_t, cast=True] keep = np.empty(n, dtype='bool')

    for i in range(n):
        if isnan(a[i]):
            v = None
        else:
            v = a[i]
        group[i] = d.get(v, -1)
        if group[i] == -1:
            group[i] = count
            d[v] = count
            count += 1

        counts[group[i]] += 1

    for i in range(n):
        keep[i] = counts[group[i]] == 1

    return keep

def unique_bool_none(ndarray[np.uint8_t, cast=True] a):
    cdef int i, n = len(a)
    cdef np.int64_t total = a.sum()

    if n == 1:
        return np.ones(1, dtype='bool')
    elif n == 2:
        if a[0] != a[1]:
            return np.ones(2, dtype='bool')
        else:
            return np.zeros(2, dtype='bool')
    else:
        if total == 1:
            return a
        elif total == n - 1:
            return ~a
        else:
            return np.zeros(n, dtype='bool')

def unique_str_none_2d(ndarray[object, ndim=2] a):
    cdef int i, j, count = 0
    cdef int n = len(a)
    cdef int nc = a.shape[1]
    cdef dict d = {}
    cdef list v = list(range(nc))
    cdef tuple t
    cdef ndarray[np.int64_t] group = np.zeros(n, dtype=np.int64)
    cdef ndarray[np.int64_t] counts = np.zeros(n, dtype=np.int64)
    cdef ndarray[np.uint8_t, cast=True] keep = np.empty(n, dtype='bool')

    for i in range(n):
        for j in range(nc):
            v[j] = a[i, j]
        t = tuple(v)

        group[i] = d.get(t, -1)
        if group[i] == -1:
            group[i] = count
            d[t] = count
            count += 1
        counts[group[i]] += 1

    for i in range(n):
        keep[i] = counts[group[i]] == 1

    return keep

def unique_bool_none_2d(ndarray[np.uint8_t, cast=True, ndim=2] a):
    cdef int i, j, count = 0
    cdef nr = a.shape[0]
    cdef nc = a.shape[1]
    cdef np.uint8_t first
    cdef ndarray[np.int64_t] group = np.empty(nr, dtype='int64')
    cdef ndarray[np.int64_t] powers
    cdef ndarray[np.int64_t] counts = np.zeros(nr, dtype='int64')
    cdef ndarray[np.uint8_t, cast = True] keep = np.empty(nr, dtype='bool')
    cdef ndarray[np.int64_t] unique
    cdef long cur_total
    cdef dict d = {}
    cdef list v = list(range(nc))

    if nc <= 20:
        unique = np.full(2 ** nc, -1, dtype='int64')
        powers = 2 ** np.arange(nc)
        for i in range(nr):
            cur_total = 0
            for j in range(nc):
                cur_total += powers[j] * a[i, j]

            group[i] = unique[cur_total]
            if group[i] == -1:
                unique[cur_total] = count
                group[i] = count
                count += 1
            counts[group[i]] += 1
    else:
        group = np.empty(nr, dtype='int64')
        for i in range(nr):
            for j in range(nc):
                v[j] = a[i, j]
            t = tuple(v)

            group[i] = d.get(t, -1)
            if group[i] == -1:
                group[i] = count
                d[t] = count
                count += 1
            counts[group[i]] += 1

    for i in range(nr):
        keep[i] = counts[group[i]] == 1

    return keep

def unique_int_none_2d(ndarray[np.int64_t, ndim=2] a):
    cdef int i, count = 0
    cdef int nr = a.shape[0]
    cdef int nc = a.shape[1]
    cdef dict d = {}
    cdef ndarray[np.uint8_t, cast = True] keep = np.empty(nr, dtype='bool')
    cdef ndarray[np.int64_t] counts = np.zeros(nr, dtype='int64')
    cdef ndarray[np.int64_t] group = np.empty(nr, dtype='int64')
    cdef long total_range = 1
    cdef long cur_range = 10 ** 7
    cdef int size = sizeof(np.int64_t) * nc
    cdef bytes string

    lows, highs = min_max_int2(a, 0)

    ranges = highs - lows + 1
    for i in range(nc):
        cur_range /= ranges[i]
        total_range *= ranges[i]

    if cur_range > 1:
        return unique_int_bounded_none_2d(a, lows, highs, ranges, total_range)

    for i in range(nr):
        string = PyBytes_FromStringAndSize(<char*>&a[i, 0], size)
        group[i] = d.get(string, -1)
        if group[i] == -1:
            group[i] = count
            d[string] = count
            count += 1
        counts[group[i]] += 1

    for i in range(nr):
        keep[i] = counts[group[i]] == 1

    return keep

def unique_float_none_2d(ndarray[np.float64_t, ndim=2] a):
    cdef int i, count = 0
    cdef int nr = a.shape[0]
    cdef int nc = a.shape[1]
    cdef dict d = {}
    cdef ndarray[np.uint8_t, cast = True] keep = np.empty(nr, dtype='bool')
    cdef ndarray[np.int64_t] counts = np.zeros(nr, dtype='int64')
    cdef ndarray[np.int64_t] group = np.empty(nr, dtype='int64')
    cdef int size = sizeof(np.float64_t) * nc
    cdef bytes string


    for i in range(nr):
        string = PyBytes_FromStringAndSize(<char*>&a[i, 0], size)
        group[i] = d.get(string, -1)
        if group[i] == -1:
            group[i] = count
            d[string] = count
            count += 1
        counts[group[i]] += 1

    for i in range(nr):
        keep[i] = counts[group[i]] == 1

    return keep

def unique_int_bounded_none_2d(ndarray[np.int64_t, ndim=2] a, ndarray[np.int64_t] lows,
                               ndarray[np.int64_t] highs, ndarray[np.int64_t] ranges, int total_range):
    cdef int i, j, count = 0
    cdef int nr = a.shape[0]
    cdef int nc = a.shape[1]

    cdef ndarray[np.int64_t, ndim=2] idx
    cdef ndarray[np.int64_t] unique = np.full(total_range, -1, dtype='int64')
    cdef ndarray[np.uint8_t, cast=True] keep = np.empty(nr, dtype='bool')
    cdef ndarray[np.int64_t] group = np.empty(nr, dtype='int64')
    cdef ndarray[np.int64_t] counts = np.zeros(nr, dtype='int64')
    cdef long iloc
    cdef ndarray[np.int64_t] range_prod

    idx = a - lows
    range_prod = np.cumprod(ranges[:nc - 1])

    for i in range(nr):
        iloc = idx[i, 0]
        for j in range(nc - 1):
            iloc += range_prod[j] * idx[i, j + 1]

        group[i] = unique[iloc]
        if group[i] == -1:
            unique[iloc] = count
            group[i] = count
            count += 1
        counts[group[i]] += 1

    for i in range(nr):
        keep[i] = counts[group[i]] == 1

    return keep

def unique_float_string_none(ndarray[np.float64_t] f, ndarray[object, ndim=2] o):
    cdef int i, j, count = 0
    cdef int nr = f.shape[0]
    cdef int nco = o.shape[1]
    cdef dict d = {}
    cdef list v = list(range(nco + 1))
    cdef ndarray[np.uint8_t, cast = True] keep = np.empty(nr, dtype='bool')
    cdef ndarray[np.int64_t] group = np.empty(nr, dtype='int64')
    cdef ndarray[np.int64_t] counts = np.zeros(nr, dtype='int64')

    for i in range(nr):
        if isnan(f[i]):
            v[0] = None
        else:
            v[0] = f[i]
        for j in range(nco):
            v[j + 1] = o[i, j]

        t = tuple(v)
        group[i] = d.get(t, -1)
        if group[i] == -1:
            group[i] = count
            d[t] = count
            count += 1
        counts[group[i]] += 1

    for i in range(nr):
        keep[i] = counts[group[i]] == 1

    return keep

def unique_int_string_none(ndarray[np.int64_t] a, ndarray[object, ndim=2] o):
    cdef int i, j, count = 0
    cdef int nr = a.shape[0]
    cdef int nco = o.shape[1]
    cdef dict d = {}
    cdef list v = list(range(nco + 1))
    cdef ndarray[np.uint8_t, cast = True] keep = np.empty(nr, dtype='bool')
    cdef ndarray[np.int64_t] group = np.empty(nr, dtype='int64')
    cdef ndarray[np.int64_t] counts = np.zeros(nr, dtype='int64')

    for i in range(nr):
        v[0] = a[i]
        for j in range(nco):
            v[j + 1] = o[i, j]

        t = tuple(v)
        group[i] = d.get(t, -1)
        if group[i] == -1:
            group[i] = count
            d[t] = count
            count += 1
        counts[group[i]] += 1

    for i in range(nr):
        keep[i] = counts[group[i]] == 1

    return keep

def unique_bool_string_none(ndarray[np.uint8_t, cast=True] a, ndarray[object, ndim=2] o):
    cdef int i, j, count = 0
    cdef int nr = a.shape[0]
    cdef int ncb = a.shape[1]
    cdef int nco = o.shape[1]
    cdef dict d = {}
    cdef list v = list(range(nco + 1))
    cdef ndarray[np.uint8_t, cast = True] keep = np.empty(nr, dtype='bool')
    cdef ndarray[np.int64_t] group = np.empty(nr, dtype='int64')
    cdef ndarray[np.int64_t] counts = np.zeros(nr, dtype='int64')

    for i in range(nr):
        v[0] = a[i]
        for j in range(nco):
            v[j + 1] = o[i, j]

        t = tuple(v)
        group[i] = d.get(t, -1)
        if group[i] == -1:
            group[i] = count
            d[t] = count
            count += 1
        counts[group[i]] += 1

    for i in range(nr):
        keep[i] = counts[group[i]] == 1

    return keep


def unique_float_string_none_2d(ndarray[np.float64_t, ndim=2] f, ndarray[object, ndim=2] o):
    cdef int i, j, count = 0
    cdef int nr = f.shape[0]
    cdef int ncf = f.shape[1]
    cdef int nco = o.shape[1]
    cdef dict d = {}
    cdef list v = list(range(nco + 1))
    cdef ndarray[np.uint8_t, cast = True] keep = np.empty(nr, dtype='bool')
    cdef ndarray[np.int64_t] group = np.empty(nr, dtype='int64')
    cdef ndarray[np.int64_t] counts = np.zeros(nr, dtype='int64')

    for i in range(nr):
        string = PyBytes_FromStringAndSize(<char*>&f[i, 0], sizeof(np.float64_t) * ncf)
        v[0] = string
        for j in range(nco):
            v[j + 1] = o[i, j]

        t = tuple(v)
        group[i] = d.get(t, -1)
        if group[i] == -1:
            group[i] = count
            d[t] = count
            count += 1
        counts[group[i]] += 1

    for i in range(nr):
        keep[i] = counts[group[i]] == 1

    return keep

def unique_int_string_none_2d(ndarray[np.int64_t, ndim=2] a, ndarray[object, ndim=2] o):
    cdef int i, j, count = 0
    cdef int nr = a.shape[0]
    cdef int nci = a.shape[1]
    cdef int nco = o.shape[1]
    cdef dict d = {}
    cdef list v = list(range(nco + 1))
    cdef ndarray[np.uint8_t, cast = True] keep = np.empty(nr, dtype='bool')
    cdef ndarray[np.int64_t] group = np.empty(nr, dtype='int64')
    cdef ndarray[np.int64_t] counts = np.zeros(nr, dtype='int64')

    for i in range(nr):
        string = PyBytes_FromStringAndSize(<char*>&a[i, 0], sizeof(np.int64_t) * nci)
        v[0] = string
        for j in range(nco):
            v[j + 1] = o[i, j]

        t = tuple(v)
        group[i] = d.get(t, -1)
        if group[i] == -1:
            group[i] = count
            d[t] = count
            count += 1
        counts[group[i]] += 1

    for i in range(nr):
        keep[i] = counts[group[i]] == 1

    return keep

def unique_bool_string_none_2d(ndarray[np.uint8_t, ndim=2, cast=True] a, ndarray[object, ndim=2] o):
    cdef int i, j, count = 0
    cdef int nr = a.shape[0]
    cdef int ncb = a.shape[1]
    cdef int nco = o.shape[1]
    cdef dict d = {}
    cdef list v = list(range(nco + 1))
    cdef ndarray[np.uint8_t, cast = True] keep = np.empty(nr, dtype='bool')
    cdef ndarray[np.int64_t] group = np.empty(nr, dtype='int64')
    cdef ndarray[np.int64_t] counts = np.zeros(nr, dtype='int64')

    for i in range(nr):
        string = PyBytes_FromStringAndSize(<char*>&a[i, 0], sizeof(np.uint8_t) * ncb)
        v[0] = string
        for j in range(nco):
            v[j + 1] = o[i, j]

        t = tuple(v)
        group[i] = d.get(t, -1)
        if group[i] == -1:
            group[i] = count
            d[t] = count
            count += 1
        counts[group[i]] += 1

    for i in range(nr):
        keep[i] = counts[group[i]] == 1

    return keep