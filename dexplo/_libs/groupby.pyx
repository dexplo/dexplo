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



def get_group_assignment(ndarray[object] a):
    cdef int i
    cdef int n = len(a)
    cdef int count = 0
    cdef ndarray[np.int64_t] group = np.empty(n, dtype=np.int64)
    cdef ndarray[object, ndim=2] group_names = np.empty((n, 1), dtype='O')
    cdef ndarray[np.int64_t] group_position = np.empty(n, dtype=np.int64)
    cdef dict d = {}

    for i in range(n):
        group[i] = d.get(a[i], -1)
        if group[i] == -1:
            group_position[count] = i
            group[i] = count
            d[a[i]] = count
            group_names[count] = a[i]
            count += 1
    return group, group_names[:count], group_position[:count]


def size(ndarray[np.int64_t] a, int group_size):
    cdef int i
    cdef int n = len(a)
    cdef ndarray[np.int64_t] result = np.zeros(group_size, dtype=np.int64)
    for i in range(n):
        result[a[i]] += 1
    return result

def count_float(ndarray[np.int64_t] labels, int size, ndarray[np.float64_t, ndim=2] data):
    cdef int i, j
    cdef nr = data.shape[0]
    cdef nc = data.shape[1]
    cdef ndarray[np.int64_t, ndim=2] result = np.zeros((size, nc), dtype='int64')
    for i in range(nc):
        for j in range(nr):
            if not isnan(data[j, i]):
                result[labels[j], i] += 1
    return result

def count_str(ndarray[np.int64_t] labels, int size, ndarray[object, ndim=2] data):
    cdef int i, j
    cdef nr = data.shape[0]
    cdef nc = data.shape[1]
    cdef ndarray[np.int64_t, ndim=2] result = np.zeros((size, nc), dtype='int64')
    for i in range(nc):
        for j in range(nr):
            if data[j, i] is not None:
                result[labels[j], i] += 1
    return result
