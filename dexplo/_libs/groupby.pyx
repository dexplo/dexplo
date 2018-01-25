import numpy as np
cimport numpy as np
from numpy cimport ndarray
from collections import defaultdict
import cython
from cpython cimport dict, set, list, tuple

@cython.wraparound(False)
@cython.boundscheck(False)
def get_group_assignment(ndarray[object] a):
    cdef int i
    cdef int n = len(a)
    cdef int count = 0
    cdef ndarray[np.int64_t] group = np.empty(n, dtype=np.int64)
    cdef ndarray[object, ndim=2] arr_names = np.empty((n, 1), dtype='O')
    cdef ndarray[np.int64_t] group_position = np.empty(n, dtype=np.int64)
    cdef dict d = {}

    for i in range(n):
        group[i] = d.get(a[i], -1)
        if group[i] == -1:
            group_position[count] = i
            group[i] = count
            d[a[i]] = count
            arr_names[count] = a[i]
            count += 1
    return group, arr_names[:count], group_position[:count]


@cython.wraparound(False)
@cython.boundscheck(False)
def size(ndarray[np.int64_t] a, int group_size):
    cdef int i
    cdef int n = len(a)
    cdef ndarray[np.int64_t] c = np.zeros(group_size, dtype=np.int64)
    for i in range(n):
        c[a[i]] += 1
    return c