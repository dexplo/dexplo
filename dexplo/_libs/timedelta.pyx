import numpy as np
from numpy import nan
from libc.math cimport ceil, round, abs
from numpy cimport ndarray
cimport numpy as np
import cython
import datetime


def seconds(ndarray[np.int64_t, ndim=2] a):
    cdef int i, j
    cdef int nr = a.shape[0]
    cdef int nc = a.shape[1]
    cdef ndarray[np.float64_t, ndim=2] result = np.empty((nr, nc), dtype='float64')
    cdef np.int64_t NAT = np.datetime64('nat').astype('int64')
    cdef long year_nanos = 10 ** 9 * 86400

    for j in range(nc):
        for i in range(nr):
            if a[i, j] == NAT:
                result[i, j] = nan
            else:
                result[i, j] = <long> (a[i, j] / 10 ** 9) % 86400
    return result


def milliseconds(ndarray[np.int64_t, ndim=2] a, total):
    cdef int i, j
    cdef int nr = a.shape[0]
    cdef int nc = a.shape[1]
    cdef ndarray[np.float64_t, ndim=2] result = np.empty((nr, nc), dtype='float64')
    cdef np.int64_t NAT = np.datetime64('nat').astype('int64')
    cdef double nanos = 10 ** 6

    if total:
        for j in range(nc):
            for i in range(nr):
                if a[i, j] == NAT:
                    result[i, j] = nan
                else:
                    result[i, j] = a[i, j] % 10 ** 9 / nanos
    else:
        for j in range(nc):
            for i in range(nr):
                if a[i, j] == NAT:
                    result[i, j] = nan
                else:
                    result[i, j] = <long> (a[i, j] % 10 ** 9) / 10 ** 6
    return result

def microseconds(ndarray[np.int64_t, ndim=2] a, total):
    cdef int i, j
    cdef int nr = a.shape[0]
    cdef int nc = a.shape[1]
    cdef ndarray[np.float64_t, ndim=2] result = np.empty((nr, nc), dtype='float64')
    cdef np.int64_t NAT = np.datetime64('nat').astype('int64')
    cdef double nanos = 10 ** 3

    if total:
        for j in range(nc):
            for i in range(nr):
                if a[i, j] == NAT:
                    result[i, j] = nan
                else:
                    result[i, j] = a[i, j] % 10 ** 9 / nanos
    else:
        for j in range(nc):
            for i in range(nr):
                if a[i, j] == NAT:
                    result[i, j] = nan
                else:
                    result[i, j] = <long> (a[i, j] % 10 ** 9) / 10 ** 3
    return result

def nanoseconds(ndarray[np.int64_t, ndim=2] a):
    cdef int i, j
    cdef int nr = a.shape[0]
    cdef int nc = a.shape[1]
    cdef ndarray[np.float64_t, ndim=2] result = np.empty((nr, nc), dtype='float64')
    cdef np.int64_t NAT = np.datetime64('nat').astype('int64')
    cdef long year_nanos = 10 ** 9 * 86400

    for j in range(nc):
        for i in range(nr):
            if a[i, j] == NAT:
                result[i, j] = nan
            else:
                result[i, j] = <long> (a[i, j] % 10 ** 9)
    return result

def days(ndarray[np.int64_t, ndim=2] a, total):
    cdef int i, j
    cdef int nr = a.shape[0]
    cdef int nc = a.shape[1]
    cdef ndarray[np.float64_t, ndim=2] result = np.empty((nr, nc), dtype='float64')
    cdef np.int64_t NAT = np.datetime64('nat').astype('int64')
    cdef double nanos = 10 ** 9 * 86400

    if total:
        for j in range(nc):
            for i in range(nr):
                if a[i, j] == NAT:
                    result[i, j] = nan
                else:
                    result[i, j] = a[i, j] / nanos
    else:
        for j in range(nc):
            for i in range(nr):
                if a[i, j] == NAT:
                    result[i, j] = nan
                else:
                    result[i, j] = <long> (a[i, j] / nanos)
    return result