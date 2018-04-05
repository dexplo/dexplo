import numpy as np
cimport numpy as np
from numpy cimport ndarray
from numpy import nan

NaT = np.datetime64('nat')

def pivot_bool(ndarray[np.int64_t] row_idx, Py_ssize_t nr,
               ndarray[np.int64_t] col_idx, Py_ssize_t nc,
               ndarray[np.uint8_t, cast=True] values):
    cdef:
        Py_ssize_t i, n = len(row_idx), cur_i, cur_j
        ndarray[np.uint8_t, ndim=2, cast=True] result = np.empty((nr, nc), dtype='bool', order='F')

    for i in range(n):
        cur_i = row_idx[i]
        cur_j = col_idx[i]
        result[cur_i, cur_j] = values[i]

    return result

def pivot_int(ndarray[np.int64_t] row_idx, Py_ssize_t nr,
              ndarray[np.int64_t] col_idx, Py_ssize_t nc,
              ndarray[np.int64_t] values):
    cdef:
        Py_ssize_t i, n = len(row_idx), cur_i, cur_j
        ndarray[np.int64_t, ndim=2] result = np.empty((nr, nc), dtype='int64', order='F')

    for i in range(n):
        cur_i = row_idx[i]
        cur_j = col_idx[i]
        result[cur_i, cur_j] = values[i]

    return result

def pivot_float(ndarray[np.int64_t] row_idx, Py_ssize_t nr,
                ndarray[np.int64_t] col_idx, Py_ssize_t nc,
                ndarray[np.float64_t] values):
    cdef:
        Py_ssize_t i, n = len(row_idx), cur_i, cur_j
        ndarray[np.float64_t, ndim=2] result = np.full((nr, nc), nan, dtype='float64', order='F')

    for i in range(n):
        cur_i = row_idx[i]
        cur_j = col_idx[i]
        result[cur_i, cur_j] = values[i]

    return result

def pivot_str(ndarray[np.int64_t] row_idx, Py_ssize_t nr,
              ndarray[np.int64_t] col_idx, Py_ssize_t nc,
              ndarray[object] values):
    cdef:
        Py_ssize_t i, n = len(row_idx), cur_i, cur_j
        ndarray[object, ndim=2] result = np.empty((nr, nc), dtype='O', order='F')

    for i in range(n):
        cur_i = row_idx[i]
        cur_j = col_idx[i]
        result[cur_i, cur_j] = values[i]

    return result

def pivot_datetime(ndarray[np.int64_t] row_idx, Py_ssize_t nr,
                   ndarray[np.int64_t] col_idx, Py_ssize_t nc,
                   ndarray[np.int64_t] values):
    cdef:
        Py_ssize_t i, n = len(row_idx), cur_i, cur_j
        ndarray[np.int64_t, ndim=2] result = np.full((nr, nc), NaT, dtype='int64', order='F')

    for i in range(n):
        cur_i = row_idx[i]
        cur_j = col_idx[i]
        result[cur_i, cur_j] = values[i]

    return result.astype('datetime64[ns]')

def pivot_timedelta(ndarray[np.int64_t] row_idx, Py_ssize_t nr,
                   ndarray[np.int64_t] col_idx, Py_ssize_t nc,
                   ndarray[np.int64_t] values):
    cdef:
        Py_ssize_t i, n = len(row_idx), cur_i, cur_j
        ndarray[np.int64_t, ndim=2] result = np.full((nr, nc), NaT, dtype='int64', order='F')

    for i in range(n):
        cur_i = row_idx[i]
        cur_j = col_idx[i]
        result[cur_i, cur_j] = values[i]

    return result.astype('timedelta64[ns]')
