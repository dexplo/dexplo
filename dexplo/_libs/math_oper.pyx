#cython: cdivision=False
#asdfcython: boundscheck=False
#asfdcython: wraparound=False
import numpy as np
cimport numpy as np
from numpy cimport ndarray
from libc.math cimport isnan

try:
    import bottleneck as bn
except ImportError:
    import numpy as bn


cdef np.float64_t MAX_FLOAT = np.finfo(np.float64).max
cdef np.float64_t MIN_FLOAT = np.finfo(np.float64).min

cdef np.int64_t MAX_INT = np.iinfo(np.int64).max
cdef np.int64_t MIN_INT = np.iinfo(np.int64).min

# int operation int

def int_add_int(ndarray[np.int64_t, ndim=2] a, int other):
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

def int_radd_int(ndarray[np.int64_t, ndim=2] a, int other):
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

def int_sub_int(ndarray[np.int64_t, ndim=2] a, int other):
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

def int_rsub_int(ndarray[np.int64_t, ndim=2] a, int other):
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
                a_new[j, i] = other - a[j, i]
    return a_new

def int_mul_int(ndarray[np.int64_t, ndim=2] a, int other):
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

def int_rmul_int(ndarray[np.int64_t, ndim=2] a, int other):
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

def int_truediv_int(ndarray[np.int64_t, ndim=2] a, int other):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0]
        int nc = a.shape[1]
        ndarray[np.float64_t, ndim=2] a_new = np.empty((nr, nc), dtype='float64')

    for i in range(nc):
        for j in range(nr):
            if a[j, i] == MIN_INT:
                a_new[j, i] = np.nan
            elif other == 0:
                if a[j, i] < 0:
                    a_new[j, i] = -np.inf
                elif a[j, i] > 0:
                    a_new[j, i] = np.inf
                else:
                    a_new[j, i] = np.nan
            else:
                a_new[j, i] = a[j, i] / other
    return a_new

def int_rtruediv_int(ndarray[np.int64_t, ndim=2] a, int other):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0]
        int nc = a.shape[1]
        ndarray[np.float64_t, ndim=2] a_new = np.empty((nr, nc), dtype='float64')

    for i in range(nc):
        for j in range(nr):
            if a[j, i] == MIN_INT:
                a_new[j, i] = np.nan
            elif a[j, i] == 0:
                if other < 0:
                    a_new[j, i] = -np.inf
                elif other > 0:
                    a_new[j, i] = np.inf
                else:
                    a_new[j, i] = np.nan
            else:
                a_new[j, i] = other / a[j, i]
    return a_new

def int_floordiv_int(ndarray[np.int64_t, ndim=2] a, int other):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0]
        int nc = a.shape[1]
        ndarray[np.int64_t, ndim=2] a_new = np.empty((nr, nc), dtype='int64')

    for i in range(nc):
        for j in range(nr):
            if a[j, i] == MIN_INT or other == 0:
                a_new[j, i] = MIN_INT
            else:
                a_new[j, i] = a[j, i] // other
    return a_new

def int_rfloordiv_int(ndarray[np.int64_t, ndim=2] a, int other):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0]
        int nc = a.shape[1]
        ndarray[np.int64_t, ndim=2] a_new = np.empty((nr, nc), dtype='int64')

    for i in range(nc):
        for j in range(nr):
            if a[j, i] == MIN_INT or a[j, i] == 0:
                a_new[j, i] = MIN_INT
            else:
                a_new[j, i] = other // a[j, i]
    return a_new

def int_pow_int(ndarray[np.int64_t, ndim=2] a, int other):
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
                a_new[j, i] = a[j, i] ** other
    return a_new

def int_rpow_int(ndarray[np.int64_t, ndim=2] a, int other):
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
                a_new[j, i] = other ** a[j, i]
    return a_new

def int_mod_int(ndarray[np.int64_t, ndim=2] a, int other):
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
                a_new[j, i] = a[j, i] % other
    return a_new

def int_rmod_int(ndarray[np.int64_t, ndim=2] a, int other):
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
                a_new[j, i] = other % a[j, i]
    return a_new

# int operation float

def int_add_float(ndarray[np.int64_t, ndim=2] a, np.float64_t other):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0]
        int nc = a.shape[1]
        ndarray[np.float64_t, ndim=2] a_new = np.empty((nr, nc), dtype='float64')

    for i in range(nc):
        for j in range(nr):
            if a[j, i] == MIN_INT:
                a_new[j, i] = np.nan
            else:
                a_new[j, i] = a[j, i] + other
    return a_new

def int_radd_float(ndarray[np.int64_t, ndim=2] a, np.float64_t other):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0]
        int nc = a.shape[1]
        ndarray[np.float64_t, ndim=2] a_new = np.empty((nr, nc), dtype='float64')

    for i in range(nc):
        for j in range(nr):
            if a[j, i] == MIN_INT:
                a_new[j, i] = np.nan
            else:
                a_new[j, i] = a[j, i] + other
    return a_new

def int_sub_float(ndarray[np.int64_t, ndim=2] a, np.float64_t other):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0]
        int nc = a.shape[1]
        ndarray[np.float64_t, ndim=2] a_new = np.empty((nr, nc), dtype='float64')

    for i in range(nc):
        for j in range(nr):
            if a[j, i] == MIN_INT:
                a_new[j, i] = np.nan
            else:
                a_new[j, i] = a[j, i] - other
    return a_new

def int_rsub_float(ndarray[np.int64_t, ndim=2] a, np.float64_t other):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0]
        int nc = a.shape[1]
        ndarray[np.float64_t, ndim=2] a_new = np.empty((nr, nc), dtype='float64')

    for i in range(nc):
        for j in range(nr):
            if a[j, i] == MIN_INT:
                a_new[j, i] = np.nan
            else:
                a_new[j, i] = other - a[j, i]
    return a_new

def int_mul_float(ndarray[np.int64_t, ndim=2] a, np.float64_t other):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0]
        int nc = a.shape[1]
        ndarray[np.float64_t, ndim=2] a_new = np.empty((nr, nc), dtype='float64')

    for i in range(nc):
        for j in range(nr):
            if a[j, i] == MIN_INT:
                a_new[j, i] = np.nan
            else:
                a_new[j, i] = a[j, i] * other
    return a_new

def int_rmul_float(ndarray[np.int64_t, ndim=2] a, np.float64_t other):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0]
        int nc = a.shape[1]
        ndarray[np.float64_t, ndim=2] a_new = np.empty((nr, nc), dtype='float64')

    for i in range(nc):
        for j in range(nr):
            if a[j, i] == MIN_INT:
                a_new[j, i] = np.nan
            else:
                a_new[j, i] = a[j, i] * other
    return a_new

def int_truediv_float(ndarray[np.int64_t, ndim=2] a, np.float64_t other):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0]
        int nc = a.shape[1]
        ndarray[np.float64_t, ndim=2] a_new = np.empty((nr, nc), dtype='float64')

    for i in range(nc):
        for j in range(nr):
            if a[j, i] == MIN_INT:
                a_new[j, i] = np.nan
            elif other == 0:
                if a[j, i] < 0:
                    a_new[j, i] = -np.inf
                elif a[j, i] > 0:
                    a_new[j, i] = np.inf
                else:
                    a_new[j, i] = np.nan
            else:
                a_new[j, i] = a[j, i] / other
    return a_new

def int_rtruediv_float(ndarray[np.int64_t, ndim=2] a, np.float64_t other):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0]
        int nc = a.shape[1]
        ndarray[np.float64_t, ndim=2] a_new = np.empty((nr, nc), dtype='float64')

    for i in range(nc):
        for j in range(nr):
            if a[j, i] == MIN_INT:
                a_new[j, i] = np.nan
            elif a[j, i] == 0:
                if other < 0:
                    a_new[j, i] = -np.inf
                elif other > 0:
                    a_new[j, i] = np.inf
                else:
                    a_new[j, i] = np.nan
            else:
                a_new[j, i] = other / a[j, i]
    return a_new

def int_floordiv_float(ndarray[np.int64_t, ndim=2] a, np.float64_t other):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0]
        int nc = a.shape[1]
        ndarray[np.float64_t, ndim=2] a_new = np.empty((nr, nc), dtype='float64')

    for i in range(nc):
        for j in range(nr):
            if a[j, i] == MIN_INT:
                a_new[j, i] = np.nan
            elif other == 0:
                if a[j, i] < 0:
                    a_new[j, i] = -np.inf
                elif a[j, i] > 0:
                    a_new[j, i] = np.inf
                else:
                    a_new[j, i] = np.nan
            else:
                a_new[j, i] = a[j, i] // other
    return a_new

def int_rfloordiv_float(ndarray[np.int64_t, ndim=2] a, np.float64_t other):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0]
        int nc = a.shape[1]
        ndarray[np.float64_t, ndim=2] a_new = np.empty((nr, nc), dtype='float64')

    for i in range(nc):
        for j in range(nr):
            if a[j, i] == MIN_INT:
                a_new[j, i] = np.nan
            elif a[j, i] == 0:
                if other < 0:
                    a_new[j, i] = -np.inf
                elif other > 0:
                    a_new[j, i] = np.inf
                else:
                    a_new[j, i] = np.nan
            else:
                a_new[j, i] = other // a[j, i]
    return a_new

def int_pow_float(ndarray[np.int64_t, ndim=2] a, np.float64_t other):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0]
        int nc = a.shape[1]
        ndarray[np.float64_t, ndim=2] a_new = np.empty((nr, nc), dtype='float64')

    for i in range(nc):
        for j in range(nr):
            if a[j, i] == MIN_INT:
                a_new[j, i] = np.nan
            else:
                a_new[j, i] = a[j, i] ** other
    return a_new

def int_rpow_float(ndarray[np.int64_t, ndim=2] a, np.float64_t other):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0]
        int nc = a.shape[1]
        ndarray[np.float64_t, ndim=2] a_new = np.empty((nr, nc), dtype='float64')

    for i in range(nc):
        for j in range(nr):
            if a[j, i] == MIN_INT:
                a_new[j, i] = np.nan
            else:
                a_new[j, i] = other ** a[j, i]
    return a_new

def int_mod_float(ndarray[np.int64_t, ndim=2] a, np.float64_t other):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0]
        int nc = a.shape[1]
        ndarray[np.float64_t, ndim=2] a_new = np.empty((nr, nc), dtype='float64')

    for i in range(nc):
        for j in range(nr):
            if a[j, i] == MIN_INT:
                a_new[j, i] = np.nan
            else:
                a_new[j, i] = a[j, i] % other
    return a_new

def int_rmod_float(ndarray[np.int64_t, ndim=2] a, np.float64_t other):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0]
        int nc = a.shape[1]
        ndarray[np.float64_t, ndim=2] a_new = np.empty((nr, nc), dtype='float64')

    for i in range(nc):
        for j in range(nr):
            if a[j, i] == MIN_INT:
                a_new[j, i] = np.nan
            else:
                a_new[j, i] = other % a[j, i]
    return a_new


# bool operation int 

def bool_add_int(ndarray[np.int8_t, ndim=2] a, int other):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0]
        int nc = a.shape[1]
        ndarray[np.int64_t, ndim=2] a_new = np.empty((nr, nc), dtype='int64')

    for i in range(nc):
        for j in range(nr):
            if a[j, i] == -1:
                a_new[j, i] = MIN_INT
            else:
                a_new[j, i] = a[j, i] + other
    return a_new

def bool_radd_int(ndarray[np.int8_t, ndim=2] a, int other):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0]
        int nc = a.shape[1]
        ndarray[np.int64_t, ndim=2] a_new = np.empty((nr, nc), dtype='int64')

    for i in range(nc):
        for j in range(nr):
            if a[j, i] == -1:
                a_new[j, i] = MIN_INT
            else:
                a_new[j, i] = a[j, i] + other
    return a_new

def bool_sub_int(ndarray[np.int8_t, ndim=2] a, int other):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0]
        int nc = a.shape[1]
        ndarray[np.int64_t, ndim=2] a_new = np.empty((nr, nc), dtype='int64')

    for i in range(nc):
        for j in range(nr):
            if a[j, i] == -1:
                a_new[j, i] = MIN_INT
            else:
                a_new[j, i] = a[j, i] - other
    return a_new

def bool_rsub_int(ndarray[np.int8_t, ndim=2] a, int other):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0]
        int nc = a.shape[1]
        ndarray[np.int64_t, ndim=2] a_new = np.empty((nr, nc), dtype='int64')

    for i in range(nc):
        for j in range(nr):
            if a[j, i] == -1:
                a_new[j, i] = MIN_INT
            else:
                a_new[j, i] = other - a[j, i]
    return a_new

def bool_mul_int(ndarray[np.int8_t, ndim=2] a, int other):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0]
        int nc = a.shape[1]
        ndarray[np.int64_t, ndim=2] a_new = np.empty((nr, nc), dtype='int64')

    for i in range(nc):
        for j in range(nr):
            if a[j, i] == -1:
                a_new[j, i] = MIN_INT
            else:
                a_new[j, i] = a[j, i] * other
    return a_new

def bool_rmul_int(ndarray[np.int8_t, ndim=2] a, int other):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0]
        int nc = a.shape[1]
        ndarray[np.int64_t, ndim=2] a_new = np.empty((nr, nc), dtype='int64')

    for i in range(nc):
        for j in range(nr):
            if a[j, i] == -1:
                a_new[j, i] = MIN_INT
            else:
                a_new[j, i] = a[j, i] * other
    return a_new

def bool_truediv_int(ndarray[np.int8_t, ndim=2] a, int other):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0]
        int nc = a.shape[1]
        ndarray[np.float64_t, ndim=2] a_new = np.empty((nr, nc), dtype='float64')

    for i in range(nc):
        for j in range(nr):
            if a[j, i] == -1:
                a_new[j, i] = np.nan
            elif other == 0:
                if a[j, i]:
                    a_new[j, i] = np.inf
                else:
                    a_new[j, i] = np.nan
            else:
                a_new[j, i] = a[j, i] / other
    return a_new

def bool_rtruediv_int(ndarray[np.int8_t, ndim=2] a, int other):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0]
        int nc = a.shape[1]
        ndarray[np.float64_t, ndim=2] a_new = np.empty((nr, nc), dtype='float64')

    for i in range(nc):
        for j in range(nr):
            if a[j, i] == -1:
                a_new[j, i] = np.nan
            elif not a[j, i]:
                if other < 0:
                    a_new[j, i] = -np.inf
                elif other > 0:
                    a_new[j, i] = np.inf
                else:
                    a_new[j, i] = np.nan
            else:
                a_new[j, i] = other
    return a_new

def bool_floordiv_int(ndarray[np.int8_t, ndim=2] a, int other):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0]
        int nc = a.shape[1]
        ndarray[np.int64_t, ndim=2] a_new = np.empty((nr, nc), dtype='int64')

    for i in range(nc):
        for j in range(nr):
            if a[j, i] == -1 or other == 0:
                a_new[j, i] = MIN_INT
            else:
                a_new[j, i] = a[j, i] // other
    return a_new

def bool_rfloordiv_int(ndarray[np.int8_t, ndim=2] a, int other):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0]
        int nc = a.shape[1]
        ndarray[np.int64_t, ndim=2] a_new = np.empty((nr, nc), dtype='int64')

    for i in range(nc):
        for j in range(nr):
            if a[j, i] == -1 or a[j, i] == 0:
                a_new[j, i] = MIN_INT
            else:
                a_new[j, i] = other // a[j, i]
    return a_new

def bool_pow_int(ndarray[np.int8_t, ndim=2] a, int other):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0]
        int nc = a.shape[1]
        ndarray[np.int64_t, ndim=2] a_new = np.empty((nr, nc), dtype='int64')

    for i in range(nc):
        for j in range(nr):
            if a[j, i] == -1:
                a_new[j, i] = MIN_INT
            else:
                a_new[j, i] = a[j, i] ** other
    return a_new

def bool_rpow_int(ndarray[np.int8_t, ndim=2] a, int other):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0]
        int nc = a.shape[1]
        ndarray[np.int64_t, ndim=2] a_new = np.empty((nr, nc), dtype='int64')

    for i in range(nc):
        for j in range(nr):
            if a[j, i] == -1:
                a_new[j, i] = MIN_INT
            else:
                a_new[j, i] = other ** a[j, i]
    return a_new

def bool_mod_int(ndarray[np.int8_t, ndim=2] a, int other):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0]
        int nc = a.shape[1]
        ndarray[np.int64_t, ndim=2] a_new = np.empty((nr, nc), dtype='int64')

    for i in range(nc):
        for j in range(nr):
            if a[j, i] == -1:
                a_new[j, i] = MIN_INT
            else:
                a_new[j, i] = a[j, i] % other
    return a_new

def bool_rmod_int(ndarray[np.int8_t, ndim=2] a, int other):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0]
        int nc = a.shape[1]
        ndarray[np.int64_t, ndim=2] a_new = np.empty((nr, nc), dtype='int64')

    for i in range(nc):
        for j in range(nr):
            if a[j, i] == -1:
                a_new[j, i] = MIN_INT
            else:
                a_new[j, i] = other % a[j, i]
    return a_new

# bool operation float

def bool_add_float(ndarray[np.int8_t, ndim=2] a, np.float64_t other):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0]
        int nc = a.shape[1]
        ndarray[np.float64_t, ndim=2] a_new = np.empty((nr, nc), dtype='float64')

    for i in range(nc):
        for j in range(nr):
            if a[j, i] == -1:
                a_new[j, i] = np.nan
            else:
                a_new[j, i] = a[j, i] + other
    return a_new

def bool_radd_float(ndarray[np.int8_t, ndim=2] a, np.float64_t other):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0]
        int nc = a.shape[1]
        ndarray[np.float64_t, ndim=2] a_new = np.empty((nr, nc), dtype='float64')

    for i in range(nc):
        for j in range(nr):
            if a[j, i] == -1:
                a_new[j, i] = np.nan
            else:
                a_new[j, i] = a[j, i] + other
    return a_new

def bool_sub_float(ndarray[np.int8_t, ndim=2] a, np.float64_t other):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0]
        int nc = a.shape[1]
        ndarray[np.float64_t, ndim=2] a_new = np.empty((nr, nc), dtype='float64')

    for i in range(nc):
        for j in range(nr):
            if a[j, i] == -1:
                a_new[j, i] = np.nan
            else:
                a_new[j, i] = a[j, i] - other
    return a_new

def bool_rsub_float(ndarray[np.int8_t, ndim=2] a, np.float64_t other):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0]
        int nc = a.shape[1]
        ndarray[np.float64_t, ndim=2] a_new = np.empty((nr, nc), dtype='float64')

    for i in range(nc):
        for j in range(nr):
            if a[j, i] == -1:
                a_new[j, i] = np.nan
            else:
                a_new[j, i] = other - a[j, i]
    return a_new

def bool_mul_float(ndarray[np.int8_t, ndim=2] a, np.float64_t other):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0]
        int nc = a.shape[1]
        ndarray[np.float64_t, ndim=2] a_new = np.empty((nr, nc), dtype='float64')

    for i in range(nc):
        for j in range(nr):
            if a[j, i] == -1:
                a_new[j, i] = np.nan
            else:
                a_new[j, i] = a[j, i] * other
    return a_new

def bool_rmul_float(ndarray[np.int8_t, ndim=2] a, np.float64_t other):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0]
        int nc = a.shape[1]
        ndarray[np.float64_t, ndim=2] a_new = np.empty((nr, nc), dtype='float64')

    for i in range(nc):
        for j in range(nr):
            if a[j, i] == -1:
                a_new[j, i] = np.nan
            else:
                a_new[j, i] = a[j, i] * other
    return a_new

def bool_truediv_float(ndarray[np.int8_t, ndim=2] a, np.float64_t other):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0]
        int nc = a.shape[1]
        ndarray[np.float64_t, ndim=2] a_new = np.empty((nr, nc), dtype='float64')

    for i in range(nc):
        for j in range(nr):
            if a[j, i] == -1:
                a_new[j, i] = np.nan
            elif other == 0:
                if a[j, i]:
                    a_new[j, i] = np.inf
                else:
                    a_new[j, i] = np.nan
            else:
                a_new[j, i] = a[j, i] / other
    return a_new

def bool_rtruediv_float(ndarray[np.int8_t, ndim=2] a, np.float64_t other):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0]
        int nc = a.shape[1]
        ndarray[np.float64_t, ndim=2] a_new = np.empty((nr, nc), dtype='float64')

    for i in range(nc):
        for j in range(nr):
            if a[j, i] == -1:
                a_new[j, i] = np.nan
            else:
                a_new[j, i] = other / a[j, i]
    return a_new

def bool_floordiv_float(ndarray[np.int8_t, ndim=2] a, np.float64_t other):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0]
        int nc = a.shape[1]
        ndarray[np.float64_t, ndim=2] a_new = np.empty((nr, nc), dtype='float64')

    for i in range(nc):
        for j in range(nr):
            if a[j, i] == -1:
                a_new[j, i] = np.nan
            elif other == 0:
                if a[j, i]:
                    a_new[j, i] = np.inf
                else:
                    a_new[j, i] = np.nan
            else:
                a_new[j, i] = a[j, i] // other
    return a_new

def bool_rfloordiv_float(ndarray[np.int8_t, ndim=2] a, np.float64_t other):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0]
        int nc = a.shape[1]
        ndarray[np.float64_t, ndim=2] a_new = np.empty((nr, nc), dtype='float64')

    for i in range(nc):
        for j in range(nr):
            if a[j, i] == -1:
                a_new[j, i] = np.nan
            elif a[j, i] == 0:
                if other < 0:
                    a_new[j, i] = -np.inf
                elif other > 0:
                    a_new[j, i] = np.inf
                else:
                    a_new[j, i] = np.nan
            else:
                a_new[j, i] = other // a[j, i]
    return a_new

def bool_pow_float(ndarray[np.int8_t, ndim=2] a, np.float64_t other):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0]
        int nc = a.shape[1]
        ndarray[np.float64_t, ndim=2] a_new = np.empty((nr, nc), dtype='float64')

    for i in range(nc):
        for j in range(nr):
            if a[j, i] == -1 or isnan(other):
                a_new[j, i] = np.nan
            else:
                a_new[j, i] = a[j, i] ** other
    return a_new

def bool_rpow_float(ndarray[np.int8_t, ndim=2] a, np.float64_t other):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0]
        int nc = a.shape[1]
        ndarray[np.float64_t, ndim=2] a_new = np.empty((nr, nc), dtype='float64')

    for i in range(nc):
        for j in range(nr):
            if a[j, i] == -1 or isnan(other):
                a_new[j, i] = np.nan
            else:
                a_new[j, i] = other ** a[j, i]
    return a_new

def bool_mod_float(ndarray[np.int8_t, ndim=2] a, np.float64_t other):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0]
        int nc = a.shape[1]
        ndarray[np.float64_t, ndim=2] a_new = np.empty((nr, nc), dtype='float64')

    for i in range(nc):
        for j in range(nr):
            if a[j, i] == -1:
                a_new[j, i] = np.nan
            else:
                a_new[j, i] = a[j, i] % other
    return a_new

def bool_rmod_float(ndarray[np.int8_t, ndim=2] a, np.float64_t other):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0]
        int nc = a.shape[1]
        ndarray[np.float64_t, ndim=2] a_new = np.empty((nr, nc), dtype='float64')

    for i in range(nc):
        for j in range(nr):
            if a[j, i] == -1:
                a_new[j, i] = np.nan
            else:
                a_new[j, i] = other % a[j, i]
    return a_new

# int operation bool

def int_add_bool(ndarray[np.int64_t, ndim=2] a, np.int8_t other):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0]
        int nc = a.shape[1]
        ndarray[np.int64_t, ndim=2] a_new = np.empty((nr, nc), dtype='int64')

    for i in range(nc):
        for j in range(nr):
            if a[j, i] == MIN_INT or other == -1:
                a_new[j, i] = MIN_INT
            else:
                a_new[j, i] = a[j, i] + other
    return a_new

def int_radd_bool(ndarray[np.int64_t, ndim=2] a, np.int8_t other):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0]
        int nc = a.shape[1]
        ndarray[np.int64_t, ndim=2] a_new = np.empty((nr, nc), dtype='int64')

    for i in range(nc):
        for j in range(nr):
            if a[j, i] == MIN_INT or other == -1:
                a_new[j, i] = MIN_INT
            else:
                a_new[j, i] = a[j, i] + other
    return a_new

def int_sub_bool(ndarray[np.int64_t, ndim=2] a, np.int8_t other):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0]
        int nc = a.shape[1]
        ndarray[np.int64_t, ndim=2] a_new = np.empty((nr, nc), dtype='int64')

    for i in range(nc):
        for j in range(nr):
            if a[j, i] == MIN_INT or other == -1:
                a_new[j, i] = MIN_INT
            else:
                a_new[j, i] = a[j, i] - other
    return a_new

def int_rsub_bool(ndarray[np.int64_t, ndim=2] a, np.int8_t other):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0]
        int nc = a.shape[1]
        ndarray[np.int64_t, ndim=2] a_new = np.empty((nr, nc), dtype='int64')

    for i in range(nc):
        for j in range(nr):
            if a[j, i] == MIN_INT or other == -1:
                a_new[j, i] = MIN_INT
            else:
                a_new[j, i] = other - a[j, i]
    return a_new

def int_mul_bool(ndarray[np.int64_t, ndim=2] a, np.int8_t other):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0]
        int nc = a.shape[1]
        ndarray[np.int64_t, ndim=2] a_new = np.empty((nr, nc), dtype='int64')

    for i in range(nc):
        for j in range(nr):
            if a[j, i] == MIN_INT or other == -1:
                a_new[j, i] = MIN_INT
            else:
                a_new[j, i] = a[j, i] * other
    return a_new

def int_rmul_bool(ndarray[np.int64_t, ndim=2] a, np.int8_t other):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0]
        int nc = a.shape[1]
        ndarray[np.int64_t, ndim=2] a_new = np.empty((nr, nc), dtype='int64')

    for i in range(nc):
        for j in range(nr):
            if a[j, i] == MIN_INT or other == -1:
                a_new[j, i] = MIN_INT
            else:
                a_new[j, i] = a[j, i] * other
    return a_new

def int_truediv_bool(ndarray[np.int64_t, ndim=2] a, np.int8_t other):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0]
        int nc = a.shape[1]
        ndarray[np.float64_t, ndim=2] a_new = np.empty((nr, nc), dtype='float64')

    for i in range(nc):
        for j in range(nr):
            if a[j, i] == MIN_INT or other == -1:
                a_new[j, i] = np.nan
            elif other == 0:
                if a[j, i] < 0:
                    a_new[j, i] = -np.inf
                elif a[j, i] > 0:
                    a_new[j, i] = np.inf
                else:
                    a_new[j, i] = np.nan
            else:
                a_new[j, i] = a[j, i] / other
    return a_new

def int_rtruediv_bool(ndarray[np.int64_t, ndim=2] a, np.int8_t other):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0]
        int nc = a.shape[1]
        ndarray[np.float64_t, ndim=2] a_new = np.empty((nr, nc), dtype='float64')

    for i in range(nc):
        for j in range(nr):
            if a[j, i] == MIN_INT or other == -1:
                a_new[j, i] = np.nan
            elif a[j, i] == 0:
                if other > 0:
                    a_new[j, i] = np.inf
                else:
                    a_new[j, i] = np.nan
            else:
                a_new[j, i] = other / a[j, i]
    return a_new

def int_floordiv_bool(ndarray[np.int64_t, ndim=2] a, np.int8_t other):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0]
        int nc = a.shape[1]
        ndarray[np.int64_t, ndim=2] a_new = np.empty((nr, nc), dtype='int64')

    for i in range(nc):
        for j in range(nr):
            if a[j, i] == MIN_INT or other == -1 or other == 0:
                a_new[j, i] = MIN_INT
            else:
                a_new[j, i] = a[j, i] // other
    return a_new

def int_rfloordiv_bool(ndarray[np.int64_t, ndim=2] a, np.int8_t other):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0]
        int nc = a.shape[1]
        ndarray[np.int64_t, ndim=2] a_new = np.empty((nr, nc), dtype='int64')

    for i in range(nc):
        for j in range(nr):
            if a[j, i] == MIN_INT or a[j, i] == 0 or other == -1:
                a_new[j, i] = MIN_INT
            else:
                a_new[j, i] = other // a[j, i]
    return a_new

def int_pow_bool(ndarray[np.int64_t, ndim=2] a, np.int8_t other):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0]
        int nc = a.shape[1]
        ndarray[np.int64_t, ndim=2] a_new = np.empty((nr, nc), dtype='int64')

    for i in range(nc):
        for j in range(nr):
            if a[j, i] == MIN_INT or other == -1:
                a_new[j, i] = MIN_INT
            else:
                a_new[j, i] = a[j, i] ** other
    return a_new

def int_rpow_bool(ndarray[np.int64_t, ndim=2] a, np.int8_t other):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0]
        int nc = a.shape[1]
        ndarray[np.int64_t, ndim=2] a_new = np.empty((nr, nc), dtype='int64')

    for i in range(nc):
        for j in range(nr):
            if a[j, i] == MIN_INT or other == -1:
                a_new[j, i] = MIN_INT
            else:
                a_new[j, i] = other ** a[j, i]
    return a_new

def int_mod_bool(ndarray[np.int64_t, ndim=2] a, np.int8_t other):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0]
        int nc = a.shape[1]
        ndarray[np.int64_t, ndim=2] a_new = np.empty((nr, nc), dtype='int64')

    for i in range(nc):
        for j in range(nr):
            if a[j, i] == MIN_INT or other == -1:
                a_new[j, i] = MIN_INT
            else:
                a_new[j, i] = a[j, i] % other
    return a_new

def int_rmod_bool(ndarray[np.int64_t, ndim=2] a, np.int8_t other):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0]
        int nc = a.shape[1]
        ndarray[np.int64_t, ndim=2] a_new = np.empty((nr, nc), dtype='int64')

    for i in range(nc):
        for j in range(nr):
            if a[j, i] == MIN_INT or other == -1:
                a_new[j, i] = MIN_INT
            else:
                a_new[j, i] = other % a[j, i]
    return a_new


# bool operation bool

def bool_add_bool(ndarray[np.int8_t, ndim=2] a, np.int8_t other):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0]
        int nc = a.shape[1]
        ndarray[np.int64_t, ndim=2] a_new = np.empty((nr, nc), dtype='int64')

    for i in range(nc):
        for j in range(nr):
            if a[j, i] == -1 or other == -1:
                a_new[j, i] = MIN_INT
            else:
                a_new[j, i] = a[j, i] + other
    return a_new

def bool_radd_bool(ndarray[np.int8_t, ndim=2] a, np.int8_t other):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0]
        int nc = a.shape[1]
        ndarray[np.int64_t, ndim=2] a_new = np.empty((nr, nc), dtype='int64')

    for i in range(nc):
        for j in range(nr):
            if a[j, i] == -1 or other == -1:
                a_new[j, i] = MIN_INT
            else:
                a_new[j, i] = a[j, i] + other
    return a_new

def bool_sub_bool(ndarray[np.int8_t, ndim=2] a, np.int8_t other):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0]
        int nc = a.shape[1]
        ndarray[np.int64_t, ndim=2] a_new = np.empty((nr, nc), dtype='int64')

    for i in range(nc):
        for j in range(nr):
            if a[j, i] == -1 or other == -1:
                a_new[j, i] = MIN_INT
            else:
                a_new[j, i] = a[j, i] - other
    return a_new

def bool_rsub_bool(ndarray[np.int8_t, ndim=2] a, np.int8_t other):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0]
        int nc = a.shape[1]
        ndarray[np.int64_t, ndim=2] a_new = np.empty((nr, nc), dtype='int64')

    for i in range(nc):
        for j in range(nr):
            if a[j, i] == -1 or other == -1:
                a_new[j, i] = MIN_INT
            else:
                a_new[j, i] = other - a[j, i]
    return a_new

def bool_mul_bool(ndarray[np.int8_t, ndim=2] a, np.int8_t other):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0]
        int nc = a.shape[1]
        ndarray[np.int64_t, ndim=2] a_new = np.empty((nr, nc), dtype='int64')

    for i in range(nc):
        for j in range(nr):
            if a[j, i] == -1 or other == -1:
                a_new[j, i] = MIN_INT
            else:
                a_new[j, i] = a[j, i] * other
    return a_new

def bool_rmul_bool(ndarray[np.int8_t, ndim=2] a, np.int8_t other):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0]
        int nc = a.shape[1]
        ndarray[np.int64_t, ndim=2] a_new = np.empty((nr, nc), dtype='int64')

    for i in range(nc):
        for j in range(nr):
            if a[j, i] == -1 or other == -1:
                a_new[j, i] = MIN_INT
            else:
                a_new[j, i] = a[j, i] * other
    return a_new

def bool_truediv_bool(ndarray[np.int8_t, ndim=2] a, np.int8_t other):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0]
        int nc = a.shape[1]
        ndarray[np.float64_t, ndim=2] a_new = np.empty((nr, nc), dtype='float64')

    for i in range(nc):
        for j in range(nr):
            if a[j, i] == -1 or other == -1:
                a_new[j, i] = np.nan
            else:
                a_new[j, i] = a[j, i] / other
    return a_new

def bool_rtruediv_bool(ndarray[np.int8_t, ndim=2] a, np.int8_t other):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0]
        int nc = a.shape[1]
        ndarray[np.float64_t, ndim=2] a_new = np.empty((nr, nc), dtype='float64')

    for i in range(nc):
        for j in range(nr):
            if a[j, i] == -1 or other == -1:
                a_new[j, i] = np.nan
            else:
                a_new[j, i] = other / a[j, i]
    return a_new

def bool_floordiv_bool(ndarray[np.int8_t, ndim=2] a, np.int8_t other):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0]
        int nc = a.shape[1]
        ndarray[np.int64_t, ndim=2] a_new = np.empty((nr, nc), dtype='int64')

    for i in range(nc):
        for j in range(nr):
            if a[j, i] == -1 or other == -1 or other == 0:
                a_new[j, i] = MIN_INT
            else:
                a_new[j, i] = a[j, i] // other
    return a_new

def bool_rfloordiv_bool(ndarray[np.int8_t, ndim=2] a, np.int8_t other):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0]
        int nc = a.shape[1]
        ndarray[np.int64_t, ndim=2] a_new = np.empty((nr, nc), dtype='int64')

    for i in range(nc):
        for j in range(nr):
            if a[j, i] == -1 or other == -1 or other == 0:
                a_new[j, i] = MIN_INT
            else:
                a_new[j, i] = other // a[j, i]
    return a_new

def bool_pow_bool(ndarray[np.int8_t, ndim=2] a, np.int8_t other):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0]
        int nc = a.shape[1]
        ndarray[np.int64_t, ndim=2] a_new = np.empty((nr, nc), dtype='int64')

    for i in range(nc):
        for j in range(nr):
            if a[j, i] == -1 or other == -1:
                a_new[j, i] = MIN_INT
            else:
                a_new[j, i] = a[j, i] ** other
    return a_new

def bool_rpow_bool(ndarray[np.int8_t, ndim=2] a, np.int8_t other):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0]
        int nc = a.shape[1]
        ndarray[np.int64_t, ndim=2] a_new = np.empty((nr, nc), dtype='int64')

    for i in range(nc):
        for j in range(nr):
            if a[j, i] == -1 or other == -1:
                a_new[j, i] = MIN_INT
            else:
                a_new[j, i] = other ** a[j, i]
    return a_new

def bool_mod_bool(ndarray[np.int8_t, ndim=2] a, np.int8_t other):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0]
        int nc = a.shape[1]
        ndarray[np.int64_t, ndim=2] a_new = np.empty((nr, nc), dtype='int64')

    for i in range(nc):
        for j in range(nr):
            if a[j, i] == -1 or other == -1:
                a_new[j, i] = MIN_INT
            else:
                a_new[j, i] = a[j, i] % other
    return a_new

def bool_rmod_bool(ndarray[np.int8_t, ndim=2] a, np.int8_t other):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0]
        int nc = a.shape[1]
        ndarray[np.int64_t, ndim=2] a_new = np.empty((nr, nc), dtype='int64')

    for i in range(nc):
        for j in range(nr):
            if a[j, i] == -1 or other == -1:
                a_new[j, i] = MIN_INT
            else:
                a_new[j, i] = other % a[j, i]
    return a_new