#asdfcython: boundscheck=False
#asfdcython: wraparound=False
import numpy as np
cimport numpy as np
from numpy cimport ndarray

try:
    import bottleneck as bn
except ImportError:
    import numpy as bn


cdef np.float64_t MAX_FLOAT = np.finfo(np.float64).max
cdef np.float64_t MIN_FLOAT = np.finfo(np.float64).min

cdef np.int64_t MAX_INT = np.iinfo(np.int64).max
cdef np.int64_t MIN_INT = np.iinfo(np.int64).min

# need to write separate operations for integer operations

def add_int(ndarray[np.int64_t, ndim=2] a, int other):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0]
        int nc = a.shape[1]
        ndarray[np.int64_t, ndim=2] a_new = np.empty((nr, nc), dtype='int64')

    for i in range(nc):
        for j in range(nr):
            if a[j, i] == MIN_INT:
                a_new[j, i] = MIN_INT
            else:
                a_new[j, i] = a[j, i] + other
    return a_new

def sub_int(ndarray[np.int64_t, ndim=2] a, int other):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0]
        int nc = a.shape[1]
        ndarray[np.int64_t, ndim=2] a_new = np.empty((nr, nc), dtype='int64')

    for i in range(nc):
        for j in range(nr):
            if a[j, i] == MIN_INT:
                a_new[j, i] = MIN_INT
            else:
                a_new[j, i] = a[j, i] - other
    return a_new

def mul_int(ndarray[np.int64_t, ndim=2] a, int other):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0]
        int nc = a.shape[1]
        ndarray[np.int64_t, ndim=2] a_new = np.empty((nr, nc), dtype='int64')

    for i in range(nc):
        for j in range(nr):
            if a[j, i] == MIN_INT:
                a_new[j, i] = MIN_INT
            else:
                a_new[j, i] = a[j, i] * other
    return a_new