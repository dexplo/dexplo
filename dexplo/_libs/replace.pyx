import numpy as np
cimport numpy as np
from numpy cimport ndarray

try:
    import bottleneck as bn
except ImportError:
    import numpy as bn

cdef extern from "numpy/npy_math.h":
    bint npy_isnan(double x)


def replace_bool_with_bool(ndarray[np.uint8_t, ndim=2, cast=True] a,
                           ndarray[np.uint8_t, cast=True] to_replace,
                           ndarray[np.uint8_t, cast=True] replacements):
    cdef:
        Py_ssize_t i, j, nr = a.shape[0], nc = a.shape[1]
        Py_ssize_t k, n = replacements.shape[0]
        ndarray[np.uint8_t, ndim=2, cast=True] result = np.empty((nr, nc), dtype='bool', order='F')

    for j in range(nc):
        for i in range(nr):
            for k in range(n):
                if a[i, j] == to_replace[k]:
                    result[i, j] = replacements[k]
                    break
                else:
                    result[i, j] = a[i, j]

    return result

def replace_bool_with_int(ndarray[np.uint8_t, ndim=2, cast=True] a,
                          ndarray[np.uint8_t, cast=True] to_replace,
                          ndarray[np.int64_t] replacements):
    cdef:
        Py_ssize_t i, j, nr = a.shape[0], nc = a.shape[1]
        Py_ssize_t k, n = replacements.shape[0]
        ndarray[np.int64_t, ndim=2] result = np.empty((nr, nc), dtype='int64', order='F')

    for j in range(nc):
        for i in range(nr):
            for k in range(n):
                if a[i, j] == to_replace[k]:
                    result[i, j] = replacements[k]
                    break
                else:
                    result[i, j] = a[i, j]

    return result

def replace_bool_with_float(ndarray[np.uint8_t, ndim=2, cast=True] a,
                            ndarray[np.uint8_t, cast=True] to_replace,
                            ndarray[np.float64_t] replacements):
    cdef:
        Py_ssize_t i, j, nr = a.shape[0], nc = a.shape[1]
        Py_ssize_t k, n = replacements.shape[0]
        ndarray[np.float64_t, ndim=2] result = np.empty((nr, nc), dtype='float64', order='F')

    for j in range(nc):
        for i in range(nr):
            for k in range(n):
                if a[i, j] == to_replace[k]:
                    result[i, j] = replacements[k]
                    break
                else:
                    result[i, j] = a[i, j]

    return result

def replace_int_with_int(ndarray[np.int64_t, ndim=2] a, ndarray[np.int64_t] to_replace,
                         ndarray[np.int64_t] replacements):
    cdef:
        Py_ssize_t i, j, nr = a.shape[0], nc = a.shape[1]
        Py_ssize_t k, n = replacements.shape[0]
        ndarray[np.int64_t, ndim=2] result = np.empty((nr, nc), dtype='int64', order='F')

    for j in range(nc):
        for i in range(nr):
            for k in range(n):
                if a[i, j] == to_replace[k]:
                    result[i, j] = replacements[k]
                    break
                else:
                    result[i, j] = a[i, j]

    return result

def replace_int_with_float(ndarray[np.int64_t, ndim=2] a, ndarray[np.int64_t] to_replace,
                           ndarray[np.float64_t] replacements):
    cdef:
        Py_ssize_t i, j, nr = a.shape[0], nc = a.shape[1]
        Py_ssize_t k, n = replacements.shape[0]
        ndarray[np.float64_t, ndim=2] result = np.empty((nr, nc), dtype='float64', order='F')

    for j in range(nc):
        for i in range(nr):
            for k in range(n):
                if a[i, j] == to_replace[k]:
                    result[i, j] = replacements[k]
                    break
                else:
                    result[i, j] = a[i, j]

    return result

def replace_float_with_float(ndarray[np.float64_t, ndim=2] a, ndarray[np.float64_t] to_replace,
                             ndarray[np.float64_t] replacements):
    cdef:
        Py_ssize_t i, j, nr = a.shape[0], nc = a.shape[1]
        Py_ssize_t k, n = replacements.shape[0], nan_idx = 0
        ndarray[np.float64_t, ndim=2] result = np.empty((nr, nc), dtype='float64', order='F')
        has_nan = False

    for i in range(n):
        if npy_isnan(to_replace[i]):
            has_nan = True
            nan_idx = i

    if has_nan:
        for j in range(nc):
            for i in range(nr):
                for k in range(n):
                    if k == nan_idx:
                        if npy_isnan(a[i, j]):
                            result[i, j] = replacements[k]
                            break
                        else:
                            result[i, j] = a[i, j]
                    else:
                        if a[i, j] == to_replace[k]:
                            result[i, j] = replacements[k]
                            break
                        else:
                            result[i, j] = a[i, j]
    else:
        for j in range(nc):
            for i in range(nr):
                for k in range(n):
                    if a[i, j] == to_replace[k]:
                        result[i, j] = replacements[k]
                        break
                    else:
                        result[i, j] = a[i, j]
    return result

def replace_str_with_str(ndarray[object, ndim=2] a, ndarray[object] to_replace,
                         ndarray[object] replacements):
    cdef:
        Py_ssize_t i, j, nr = a.shape[0], nc = a.shape[1]
        Py_ssize_t k, n = replacements.shape[0]
        ndarray[object, ndim=2] result = np.empty((nr, nc), dtype='O', order='F')

    for j in range(nc):
        for i in range(nr):
            for k in range(n):
                if a[i, j] == to_replace[k]:
                    result[i, j] = replacements[k]
                    break
                else:
                    result[i, j] = a[i, j]

    return result

def replace_datetime_with_datetime(ndarray[np.int64_t, ndim=2] a, ndarray[np.int64_t] to_replace,
                                   ndarray[np.int64_t] replacements):
    cdef:
        Py_ssize_t i, j, nr = a.shape[0], nc = a.shape[1]
        Py_ssize_t k, n = replacements.shape[0]
        ndarray[np.int64_t, ndim=2] result = np.empty((nr, nc), dtype='int64', order='F')

    for j in range(nc):
        for i in range(nr):
            for k in range(n):
                if a[i, j] == to_replace[k]:
                    result[i, j] = replacements[k]
                    break
                else:
                    result[i, j] = a[i, j]

    return result.astype('datetime64[ns]')

def replace_timedelta_with_timedelta(ndarray[np.int64_t, ndim=2] a, ndarray[np.int64_t] to_replace,
                                     ndarray[np.int64_t] replacements):
    cdef:
        Py_ssize_t i, j, nr = a.shape[0], nc = a.shape[1]
        Py_ssize_t k, n = replacements.shape[0]
        ndarray[np.int64_t, ndim=2] result = np.empty((nr, nc), dtype='int64', order='F')

    for j in range(nc):
        for i in range(nr):
            for k in range(n):
                if a[i, j] == to_replace[k]:
                    result[i, j] = replacements[k]
                    break
                else:
                    result[i, j] = a[i, j]

    return result.astype('timedelta64[ns]')