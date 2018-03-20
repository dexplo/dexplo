#cython: boundscheck=False
#cython: wraparound=False
import numpy as np
cimport numpy as np
from numpy cimport ndarray

def sort_str_map(ndarray[object] a, np.uint8_t asc):
    cdef int i
    cdef int n = len(a)
    cdef set s = set()
    cdef ndarray[object] b
    for i in range(n):
        s.add(a[i])

    b = np.sort(np.array(list(s), dtype='O'))
    if not asc:
        b = b[::-1]
    return dict(zip(b, np.arange(len(b))))

def replace_str_int(ndarray[object] a, d):
    cdef int i
    cdef int n = len(a)
    cdef ndarray[np.int64_t] b = np.empty(n, dtype='int64')

    for i in range(n):
        b[i] = d[a[i]]
    return b

def count_int_ordered(ndarray[np.int64_t] a, int num):
    cdef int i, n = len(a)
    cdef ndarray[np.int64_t] counts = np.zeros(num, dtype='int64')
    for i in range(n):
        counts[a[i]] += 1
    return counts

def get_idx(ndarray[np.int64_t] a, ndarray[np.int64_t] counts):
    cdef int i, n = len(a)
    cdef ndarray[np.int64_t] idx = np.empty(n, dtype='int64')

    counts = counts.cumsum() - counts

    for i in range(n):
        idx[counts[a[i]]] = i
        counts[a[i]] += 1
    return idx

def rank_int_min(ndarray[np.int64_t, ndim=2] arg, ndarray[np.int64_t, ndim=2] act):
    cdef int i, j
    cdef nr = arg.shape[0]
    cdef nc = arg.shape[1]
    cdef int rank
    cdef ndarray[np.int64_t, ndim=2] result = np.empty((nr, nc), dtype='int64')

    for i in range(nc):
        result[arg[0, i], i] = 1
        rank = 1
        for j in range(1, nr):
            if act[arg[j, i], i] == act[arg[j - 1, i], i]:
                result[arg[j, i], i] = rank
            else:
                result[arg[j, i], i] = j + 1
                rank = j + 1
    return result

def rank_int_max(ndarray[np.int64_t, ndim=2] arg, ndarray[np.int64_t, ndim=2] act):
    cdef int i, j
    cdef nr = arg.shape[0]
    cdef nc = arg.shape[1]
    cdef int rank
    cdef ndarray[np.int64_t, ndim=2] result = np.empty((nr, nc), dtype='int64')

    for i in range(nc):
        result[arg[nr - 1, i], i] = nr
        rank = nr
        for j in range(nr - 2, -1, -1):
            if act[arg[j, i], i] == act[arg[j + 1, i], i]:
                result[arg[j, i], i] = rank
            else:
                result[arg[j, i], i] = j + 1
                rank = j + 1
    return result

def rank_float_min(ndarray[np.int64_t, ndim=2] arg, ndarray[np.float64_t, ndim=2] act):
    cdef int i, j
    cdef nr = arg.shape[0]
    cdef nc = arg.shape[1]
    cdef int rank
    cdef ndarray[np.int64_t, ndim=2] result = np.empty((nr, nc), dtype='int64')

    for i in range(nc):
        result[arg[0, i], i] = 1
        rank = 1
        for j in range(1, nr):
            if act[arg[j, i], i] == act[arg[j - 1, i], i]:
                result[arg[j, i], i] = rank
            else:
                result[arg[j, i], i] = j + 1
                rank = j + 1
    return result

def rank_float_max(ndarray[np.int64_t, ndim=2] arg, ndarray[np.float64_t, ndim=2] act):
    cdef int i, j
    cdef nr = arg.shape[0]
    cdef nc = arg.shape[1]
    cdef int rank
    cdef ndarray[np.int64_t, ndim=2] result = np.empty((nr, nc), dtype='int64')

    for i in range(nc):
        result[arg[nr - 1, i], i] = nr
        rank = nr
        for j in range(nr - 2, -1, -1):
            if act[arg[j, i], i] == act[arg[j + 1, i], i]:
                result[arg[j, i], i] = rank
            else:
                result[arg[j, i], i] = j + 1
                rank = j + 1
    return result

def rank_bool_min(ndarray[np.int64_t, ndim=2] arg, ndarray[np.uint8_t, ndim=2, cast=True] act):
    cdef int i, j
    cdef nr = arg.shape[0]
    cdef nc = arg.shape[1]
    cdef int rank
    cdef ndarray[np.int64_t, ndim=2] result = np.empty((nr, nc), dtype='int64')

    for i in range(nc):
        result[arg[0, i], i] = 1
        rank = 1
        for j in range(1, nr):
            if act[arg[j, i], i] == act[arg[j - 1, i], i]:
                result[arg[j, i], i] = rank
            else:
                result[arg[j, i], i] = j + 1
                rank = j + 1
    return result

def rank_bool_max(ndarray[np.int64_t, ndim=2] arg, ndarray[np.uint8_t, ndim=2, cast=True] act):
    cdef int i, j
    cdef nr = arg.shape[0]
    cdef nc = arg.shape[1]
    cdef int rank
    cdef ndarray[np.int64_t, ndim=2] result = np.empty((nr, nc), dtype='int64')

    for i in range(nc):
        result[arg[nr - 1, i], i] = nr
        rank = nr
        for j in range(nr - 2, -1, -1):
            if act[arg[j, i], i] == act[arg[j + 1, i], i]:
                result[arg[j, i], i] = rank
            else:
                result[arg[j, i], i] = j + 1
                rank = j + 1
    return result

def rank_date_min(ndarray[np.int64_t, ndim=2] arg, ndarray[np.int64_t, ndim=2] act):
    cdef int i, j
    cdef nr = arg.shape[0]
    cdef nc = arg.shape[1]
    cdef int rank
    cdef ndarray[np.int64_t, ndim=2] result = np.empty((nr, nc), dtype='int64')

    for i in range(nc):
        result[arg[0, i], i] = 1
        rank = 1
        for j in range(1, nr):
            if act[arg[j, i], i] == act[arg[j - 1, i], i]:
                result[arg[j, i], i] = rank
            else:
                result[arg[j, i], i] = j + 1
                rank = j + 1
    return result

def rank_int_first(ndarray[np.int64_t, ndim=2] a, ndarray[np.int64_t, ndim=2] act):
    cdef int i, j
    cdef nr = a.shape[0]
    cdef nc = a.shape[1]
    cdef int rank
    cdef ndarray[np.int64_t, ndim=2] result = np.empty((nr, nc), dtype='int64')

    for i in range(nc):
        for j in range(nr):
            result[a[j, i], i] = j + 1
    return result

def rank_float_first(ndarray[np.int64_t, ndim=2] a, ndarray[np.float64_t, ndim=2] act):
    cdef int i, j
    cdef nr = a.shape[0]
    cdef nc = a.shape[1]
    cdef int rank
    cdef ndarray[np.int64_t, ndim=2] result = np.empty((nr, nc), dtype='int64')

    for i in range(nc):
        for j in range(nr):
            result[a[j, i], i] = j + 1
    return result

def rank_bool_first(ndarray[np.int64_t, ndim=2] a, ndarray[np.uint8_t, ndim=2, cast=True] act):
    cdef int i, j
    cdef nr = a.shape[0]
    cdef nc = a.shape[1]
    cdef int rank
    cdef ndarray[np.int64_t, ndim=2] result = np.empty((nr, nc), dtype='int64')

    for i in range(nc):
        for j in range(nr):
            result[a[j, i], i] = j + 1
    return result


def rank_int_dense(ndarray[np.int64_t, ndim=2] a, ndarray[np.int64_t, ndim=2] act):
    cdef int i, j
    cdef nr = a.shape[0]
    cdef nc = a.shape[1]
    cdef int rank
    cdef ndarray[np.int64_t, ndim=2] result = np.empty((nr, nc), dtype='int64')

    for i in range(nc):
        result[a[0, i], i] = 1
        rank = 1
        for j in range(1, nr):
            if act[a[j, i], i] == act[a[j - 1, i], i]:
                result[a[j, i], i] = rank
            else:
                rank += 1
                result[a[j, i], i] = rank
    return result

def rank_float_dense(ndarray[np.int64_t, ndim=2] a, ndarray[np.float64_t, ndim=2] act):
    cdef int i, j
    cdef nr = a.shape[0]
    cdef nc = a.shape[1]
    cdef int rank
    cdef ndarray[np.int64_t, ndim=2] result = np.empty((nr, nc), dtype='int64')

    for i in range(nc):
        result[a[0, i], i] = 1
        rank = 1
        for j in range(1, nr):
            if act[a[j, i], i] == act[a[j - 1, i], i]:
                result[a[j, i], i] = rank
            else:
                rank += 1
                result[a[j, i], i] = rank
    return result

def rank_bool_dense(ndarray[np.int64_t, ndim=2] a, ndarray[np.uint8_t, ndim=2, cast=True] act):
    cdef int i, j
    cdef nr = a.shape[0]
    cdef nc = a.shape[1]
    cdef int rank
    cdef ndarray[np.int64_t, ndim=2] result = np.empty((nr, nc), dtype='int64')

    for i in range(nc):
        result[a[0, i], i] = 1
        rank = 1
        for j in range(1, nr):
            if act[a[j, i], i] == act[a[j - 1, i], i]:
                result[a[j, i], i] = rank
            else:
                rank += 1
                result[a[j, i], i] = rank
    return result

def rank_str_min(ndarray[np.int64_t, ndim=2] arg, ndarray[object, ndim=2] act):
    cdef int i, j
    cdef nr = arg.shape[0]
    cdef nc = arg.shape[1]
    cdef int rank
    cdef ndarray[np.int64_t, ndim=2] result = np.empty((nr, nc), dtype='int64')

    for i in range(nc):
        result[arg[0, i], i] = 1
        rank = 1
        for j in range(1, nr):
            if act[arg[j, i], i] == act[arg[j - 1, i], i]:
                result[arg[j, i], i] = rank
            else:
                result[arg[j, i], i] = j + 1
                rank = j + 1
    return result

def rank_str_max(ndarray[np.int64_t, ndim=2] arg, ndarray[object, ndim=2] act):
    cdef int i, j
    cdef nr = arg.shape[0]
    cdef nc = arg.shape[1]
    cdef int rank
    cdef ndarray[np.int64_t, ndim=2] result = np.empty((nr, nc), dtype='int64')

    for i in range(nc):
        result[arg[nr - 1, i], i] = nr
        rank = nr
        for j in range(nr - 2, -1, -1):
            if act[arg[j, i], i] == act[arg[j + 1, i], i]:
                result[arg[j, i], i] = rank
            else:
                result[arg[j, i], i] = j + 1
                rank = j + 1
    return result

def rank_str_first(ndarray[np.int64_t, ndim=2] a, ndarray[object, ndim=2] act):
    cdef int i, j
    cdef nr = a.shape[0]
    cdef nc = a.shape[1]
    cdef int rank
    cdef ndarray[np.int64_t, ndim=2] result = np.empty((nr, nc), dtype='int64')

    for i in range(nc):
        for j in range(nr):
            result[a[j, i], i] = j + 1
    return result

def rank_str_dense(ndarray[np.int64_t, ndim=2] a, ndarray[object, ndim=2] act):
    cdef int i, j
    cdef nr = a.shape[0]
    cdef nc = a.shape[1]
    cdef int rank
    cdef ndarray[np.int64_t, ndim=2] result = np.empty((nr, nc), dtype='int64')

    for i in range(nc):
        result[a[0, i], i] = 1
        rank = 1
        for j in range(1, nr):
            if act[a[j, i], i] == act[a[j - 1, i], i]:
                result[a[j, i], i] = rank
            else:
                rank += 1
                result[a[j, i], i] = rank
    return result

def rank_str_min_to_first(ndarray[np.int64_t, ndim=2] cur_rank, ndarray[np.int64_t, ndim=2] arg, ndarray[object, ndim=2] arr):
    cdef int i, j, add
    cdef int nr = cur_rank.shape[0]
    cdef int nc = cur_rank.shape[1]

    for i in range(nc):
        add = 1
        for j in range(nr - 2, -1, -1):
            if arr[arg[j, i], i] == arr[arg[j + 1, i], i]:
                cur_rank[arg[j, i], i] += add
                add += 1
            else:
                add = 1
    return cur_rank


def rank_int_average(ndarray[np.int64_t, ndim=2] arg, ndarray[np.int64_t, ndim=2] act):
    cdef int i, j, k
    cdef nr = arg.shape[0]
    cdef nc = arg.shape[1]
    cdef int rank, ct
    cdef np.float64_t avg
    cdef ndarray[np.float64_t, ndim=2] result = np.empty((nr, nc), dtype='float64')

    for i in range(nc):
        result[arg[0, i], i] = 1
        rank = 1
        ct = 1
        for j in range(1, nr):
            if act[arg[j, i], i] == act[arg[j - 1, i], i]:
                result[arg[j, i], i] = rank
                ct += 1
            else:
                if ct > 1:
                    avg = (j + 1) - <double> (ct + 1) / 2
                    for k in range(ct):
                        result[arg[j - k - 1, i], i] = avg
                result[arg[j, i], i] = j + 1
                rank = j + 1
                ct = 1
        if ct > 0:
            avg = (j + 2) - <double> (ct + 1) / 2
            for k in range(ct):
                result[arg[j - k, i], i] = avg
    return result

def rank_float_average(ndarray[np.int64_t, ndim=2] arg, ndarray[np.float64_t, ndim=2] act):
    cdef int i, j, k
    cdef nr = arg.shape[0]
    cdef nc = arg.shape[1]
    cdef int rank, ct
    cdef np.float64_t avg
    cdef ndarray[np.float64_t, ndim=2] result = np.empty((nr, nc), dtype='float64')

    for i in range(nc):
        result[arg[0, i], i] = 1
        rank = 1
        ct = 1
        for j in range(1, nr):
            if act[arg[j, i], i] == act[arg[j - 1, i], i]:
                result[arg[j, i], i] = rank
                ct += 1
            else:
                if ct > 1:
                    avg = (j + 1) - <double> (ct + 1) / 2
                    for k in range(ct):
                        result[arg[j - k - 1, i], i] = avg
                result[arg[j, i], i] = j + 1
                rank = j + 1
                ct = 1
        if ct > 0:
            avg = (j + 2) - <double> (ct + 1) / 2
            for k in range(ct):
                result[arg[j - k, i], i] = avg
    return result


def rank_bool_average(ndarray[np.int64_t, ndim=2] arg, ndarray[np.uint8_t, ndim=2, cast=True] act):
    cdef int i, j, k
    cdef nr = arg.shape[0]
    cdef nc = arg.shape[1]
    cdef int rank, ct
    cdef np.float64_t avg
    cdef ndarray[np.float64_t, ndim=2] result = np.empty((nr, nc), dtype='float64')

    for i in range(nc):
        result[arg[0, i], i] = 1
        rank = 1
        ct = 1
        for j in range(1, nr):
            if act[arg[j, i], i] == act[arg[j - 1, i], i]:
                result[arg[j, i], i] = rank
                ct += 1
            else:
                if ct > 1:
                    avg = (j + 1) - <double> (ct + 1) / 2
                    for k in range(ct):
                        result[arg[j - k - 1, i], i] = avg
                result[arg[j, i], i] = j + 1
                rank = j + 1
                ct = 1
        if ct > 0:
            avg = (j + 2) - <double> (ct + 1) / 2
            for k in range(ct):
                result[arg[j - k, i], i] = avg
    return result

def rank_str_average(ndarray[np.int64_t, ndim=2] arg, ndarray[object, ndim=2] act):
    cdef int i, j, k
    cdef nr = arg.shape[0]
    cdef nc = arg.shape[1]
    cdef int rank, ct
    cdef np.float64_t avg
    cdef ndarray[np.float64_t, ndim=2] result = np.empty((nr, nc), dtype='float64')

    for i in range(nc):
        result[arg[0, i], i] = 1
        rank = 1
        ct = 1
        for j in range(1, nr):
            if act[arg[j, i], i] == act[arg[j - 1, i], i]:
                result[arg[j, i], i] = rank
                ct += 1
            else:
                if ct > 1:
                    avg = (j + 1) - <double> (ct + 1) / 2
                    for k in range(ct):
                        result[arg[j - k - 1, i], i] = avg
                result[arg[j, i], i] = j + 1
                rank = j + 1
                ct = 1
        if ct > 0:
            avg = (j + 2) - <double> (ct + 1) / 2
            for k in range(ct):
                result[arg[j - k, i], i] = avg
    return result
