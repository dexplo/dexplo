#cython: boundscheck=False
#cython: wraparound=False
import numpy as np
cimport numpy as np
cimport cython
from numpy import nan
from numpy cimport ndarray
from libc.math cimport isnan


def validate_1D_object_array(ndarray[object] arr, columns):
    cdef int i
    cdef int n = len(arr)

    cur_dtype = type(arr[0])
    for i in range(n):
        if not isinstance(arr[i], cur_dtype):
            raise TypeError(f'Found mixed data in column {columns[i]}')


def maybe_convert_object_array(ndarray[object] arr, column):
    cdef int i
    cdef int n = len(arr)

    if isinstance(arr[0], (str, bytes)):
        types = (str, bytes)
    else:
        types = type(arr[0]) # TODO: make more broad. if float use floating. need to make a mapping
    
    for i in range(n):
        if not isinstance(arr[i], types):
            raise TypeError(f'Found mixed data in column {column}.')

    if types != (str, bytes):
        return arr.astype(types)
    return arr


def validate_strings_in_object_array(ndarray[object] arr, columns=None):
    """
    Make sure only unicode strings are in array of type object
    """
    cdef int i
    cdef int n = len(arr)

    for i in range(n):
        if not isinstance(arr[i], str):
            if isinstance(arr[i], bytes):
                arr[i] = arr[i].decode()
            elif arr[i] is np.nan:
                pass
            elif arr[i] is None:
                arr[i] = np.nan
            elif columns:
                raise TypeError('Array of type "object" must only contain '
                                f'strings in column {columns[i]}')
            else:
                raise TypeError('Array of type "object" must only contain '
                                'strings')
    return arr


def isnan_object(ndarray[object, ndim=2] a):
    cdef int i, j
    cdef int nr = a.shape[0]
    cdef int nc = a.shape[1]
    cdef ndarray[np.uint8_t, cast=True] hasnan = np.zeros(nc, dtype='bool')
    for i in range(nr):
        for j in range(nc):
            if a[i][j] is nan:
                hasnan[j] = True
                break
    return hasnan


def any_int(ndarray[np.int64_t] a):
    cdef int i
    cdef int n = len(a)
    for i in range(n):
        if a[i] != 0:
            return True
    return False


def any_float(ndarray[np.float64_t] a):
    cdef int i
    cdef int n = len(a)
    for i in range(n):
        if a[i] != 0 and not isnan(a[i]):
            return True
    return False


def any_bool(ndarray[np.int8_t, cast=True] a):
    cdef int i
    cdef int n = len(a)
    for i in range(n):
        if a[i] != 0:
            return True
    return False


def any_str(ndarray[object] a):
    cdef int i
    cdef int n = len(a)
    for i in range(n):
        if a[i] != '' and not a[i] is not nan:
            return True
    return False
