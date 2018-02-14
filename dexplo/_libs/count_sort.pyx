#cython: boundscheck=False
#cython: wraparound=False
import numpy as np
cimport numpy as np
from numpy cimport ndarray
from numpy import nan
import cython
from cpython cimport dict, set, list, tuple
from libc.math cimport isnan, sqrt, floor, ceil
import cmath


def sort_str_map(ndarray[object] a, asc):
    cdef int    i
    cdef int n = len(a)
    cdef set s = set()
    cdef ndarray[object] b
    for i in range(n):
        s.add(a[i])

    b = np.sort(np.array(list(s), dtype='O'))
    if not asc:
        b = b[::-1]
    return dict(zip(b, np.arange(len(b))))

def replace_str_int(ndarray[object] a, dict d):
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
    for i in range(n):
        idx[counts[a[i]] - 1] = i
        counts[a[i]] -= 1
    return idx