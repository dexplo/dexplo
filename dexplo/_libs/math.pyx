#cython: boundscheck=False
#cython: wraparound=False
import numpy as np
cimport numpy as np
from numpy cimport ndarray
from numpy import nan
from collections import defaultdict
import cython
from cpython cimport dict, set, list, tuple
from libc.math cimport isnan, sqrt

try:
    import bottleneck as bn
except ImportError:
    import numpy as bn


MAX_FLOAT = np.finfo(np.float64).max
MIN_FLOAT = np.finfo(np.float64).min

MAX_INT = np.iinfo(np.int64).max
MIN_INT = np.iinfo(np.int64).min

MAX_CHAR = chr(1_000_000)
MIN_CHAR = chr(0)

def add_obj(ndarray[object, ndim=2] arr, str other):
    cdef int i, j
    cdef int nr = arr.shape[0]
    cdef int nc = arr.shape[1]
    cdef ndarray[object, ndim=2] final = np.empty((nr, nc), dtype='O')
    for i in range(nr):
        for j in range(nc):
            try:
                final[i, j] = arr[i, j] + other
            except:
                final[i, j] = nan
    return final

def radd_obj(ndarray[object, ndim=2] arr, str other):
    cdef int i, j
    cdef int nr = arr.shape[0]
    cdef int nc = arr.shape[1]
    cdef ndarray[object, ndim=2] final = np.empty((nr, nc), dtype='O')
    for i in range(nr):
        for j in range(nc):
            try:
                final[i, j] = other + arr[i, j]
            except:
                final[i, j] = nan
    return final

def lt_obj(ndarray[object, ndim=2] arr, str other):
    cdef int i, j
    cdef int nr = arr.shape[0]
    cdef int nc = arr.shape[1]
    cdef ndarray[np.uint8_t, ndim=2, cast=True] final = np.empty((nr, nc), dtype='bool')
    for i in range(nr):
        for j in range(nc):
            try:
                final[i, j] = arr[i, j] < other
            except:
                final[i, j] = False
    return final

def le_obj(ndarray[object, ndim=2] arr, str other):
    cdef int i, j
    cdef int nr = arr.shape[0]
    cdef int nc = arr.shape[1]
    cdef ndarray[np.uint8_t, ndim=2, cast=True] final = np.empty((nr, nc), dtype='bool')
    for i in range(nr):
        for j in range(nc):
            try:
                final[i, j] = arr[i, j] <= other
            except:
                final[i, j] = False
    return final

def gt_obj(ndarray[object, ndim=2] arr, str other):
    cdef int i, j
    cdef int nr = arr.shape[0]
    cdef int nc = arr.shape[1]
    cdef ndarray[np.uint8_t, ndim=2, cast=True] final = np.empty((nr, nc), dtype='bool')
    for i in range(nr):
        for j in range(nc):
            try:
                final[i, j] = arr[i, j] > other
            except:
                final[i, j] = False
    return final

def ge_obj(ndarray[object, ndim=2] arr, str other):
    cdef int i, j
    cdef int nr = arr.shape[0]
    cdef int nc = arr.shape[1]
    cdef ndarray[np.uint8_t, ndim=2, cast=True] final = np.empty((nr, nc), dtype='bool')
    for i in range(nr):
        for j in range(nc):
            try:
                final[i, j] = arr[i, j] >= other
            except:
                final[i, j] = False
    return final

def min_max_int(ndarray[np.int64_t] a):
    cdef int i, n = len(a)
    cdef long low = a[0]
    cdef long high = a[0]
    for i in range(n):
        if a[i] < low:
            low = a[i]
        if a[i] > high:
            high = a[i]
    return low, high

def min_max_float(ndarray[np.float64_t] a):
    cdef int i, n = len(a)
    cdef np.float64_t low = a[0]
    cdef np.float64_t high = a[0]
    for i in range(n):
        if a[i] < low:
            low = a[i]
        if a[i] > high:
            high = a[i]
    return low, high

def unique_object(ndarray[object] a):
    cdef int i, len_before
    cdef int n = len(a)
    cdef set s = set()
    cdef list l = []
    for i in range(n):
        len_before = len(s)
        s.add(a[i])
        if len(s) > len_before:
            l.append(a[i])
    return np.array(l, dtype='object')

def unique_int(ndarray[np.int64_t] a):
    cdef int i, len_before
    cdef int n = len(a)
    cdef set s = set()
    cdef list l = []
    for i in range(n):
        len_before = len(s)
        s.add(a[i])
        if len(s) > len_before:
            l.append(a[i])
    return np.array(l, dtype=np.int64)

def unique_bool(ndarray[np.uint8_t, cast=True] a):
    cdef int i, n = len(a)
    cdef ndarray[np.uint8_t, cast=True] unique = np.zeros(2, dtype=bool)
    cdef list result = []
    for i in range(n):
        if not unique[a[i]]:
            unique[a[i]] = True
            result.append(a[i])
        if len(result) == 2:
            break
    return np.array(result, dtype=bool)

def unique_float(ndarray[double] a):
    cdef int i, len_before
    cdef int n = len(a)
    cdef set s = set()
    cdef list l = []
    for i in range(n):
        len_before = len(s)
        s.add(a[i])
        if len(s) > len_before:
            l.append(a[i])
    return np.array(l, dtype=np.float64)

def unique_bounded(ndarray[np.int64_t] a, long amin):
    cdef int i, n = len(a)
    cdef ndarray[np.uint8_t, cast=True] unique = np.zeros(n, dtype=bool)
    cdef list result = []
    for i in range(n):
        if not unique[a[i] - amin]:
            unique[a[i] - amin] = True
            result.append(a[i])
    return np.array(result)

def sum_int(ndarray[np.int64_t, ndim=2] a, axis, **kwargs):
    cdef long *arr = <long*> a.data
    cdef int i, j
    cdef int nc = a.shape[1]
    cdef int nr = a.shape[0]
    cdef ndarray[np.int64_t] total

    if axis == 0:
        total = np.zeros(nc, dtype=np.int64)
        for i in range(nc):
            for j in range(nr):
                total[i] += arr[i * nr + j]
    else:
        total = np.zeros(nr, dtype=np.int64)
        for i in range(nr):
            for j in range(nc):
                total[i] += arr[j * nr + i]
    return total

def sum_bool(ndarray[np.uint8_t, ndim=2, cast=True] a, axis, **kwargs):
    cdef unsigned char *arr = <unsigned char*> a.data
    cdef int i, j
    cdef int nc = a.shape[1]
    cdef int nr = a.shape[0]
    cdef ndarray[np.uint8_t] total
    if axis == 0:
        total = np.zeros(nc, dtype=np.uint8)
        for i in range(nc):
            for j in range(nr):
                total[i] += arr[i * nr + j]
    else:
        total = np.zeros(nr, dtype=np.uint8)
        for i in range(nr):
            for j in range(nc):
                total[i] += arr[j * nr + i]
    return total.astype(np.int64)

def sum_float(ndarray[np.float64_t, ndim=2] a, axis, hasnans, **kwargs):
    cdef double *arr = <double*> a.data
    cdef int i, j
    cdef int nc = a.shape[1]
    cdef int nr = a.shape[0]
    cdef long idx
    cdef ndarray[np.float64_t] total

    if axis == 0:
        total = np.zeros(nc, dtype=np.float64)
        for i in range(nc):
            if hasnans[i] is None or hasnans[i] == True:
                for j in range(nr):
                    if not isnan(arr[i * nr + j]):
                        total[i] += arr[i * nr + j]
            else:
                for j in range(nr):
                    total[i] += arr[i * nr + j]

    else:
        total = np.zeros(nr, dtype=np.float64)
        for i in range(nc):
            if hasnans[i] is None or hasnans[i] == True:
                for j in range(nr):
                    if not isnan(arr[i * nr + j]):
                        total[j] += arr[i * nr + j]
            else:
                for j in range(nr):
                    total[j] += arr[i * nr + j]
    return total

def sum_str(ndarray[object, ndim=2] a, axis, hasnans):
    cdef int i, j
    cdef int nc = a.shape[1]
    cdef int nr = a.shape[0]
    cdef ndarray[object] total
    cdef int ct

    if axis == 0:
        total = np.zeros(nc, dtype='U').astype('O')
        for i in range(nc):
            ct = 0
            if hasnans[i] is None or hasnans[i] == True:
                for j in range(nr):
                    if a[j, i] is not nan:
                        total[i] = total[i] + a[j, i]
                        ct += 1
                if ct == 0:
                    total[i] = nan
            else:
                for j in range(nr):
                    total[i] = total[i] + a[j, i]
    else:
        total = np.zeros(nr, dtype='U').astype('O')
        for i in range(nc):
            ct = 0
            if hasnans[i] is None or hasnans[i] == True:
                for j in range(nr):
                    if a[j, i] is not nan:
                        total[j] = total[j] + a[j, i]
                        ct += 1
                if ct == 0:
                    total[i] = nan
            else:
                for j in range(nr):
                    total[j] = total[j] + a[j, i]
    return total

def max_int(ndarray[np.int64_t, ndim=2] a, axis, **kwargs):
    cdef long *arr = <long*> a.data
    cdef int i, j
    cdef int nc = a.shape[1]
    cdef int nr = a.shape[0]
    cdef ndarray[np.int64_t] amax
    
    if axis ==0:
        amax = np.empty(nc, dtype=np.int64)
        amax.fill(MIN_INT)
        for i in range(nc):
            for j in range(nr):
                if arr[i * nr + j] > amax[i]:
                    amax[i] = arr[i * nr + j]
    else:
        amax = np.empty(nr, dtype=np.int64)
        amax.fill(MIN_INT)
        for i in range(nc):
            for j in range(nr):
                if arr[i * nr + j] > amax[j]:
                    amax[j] = arr[i * nr + j]
    return amax

def min_int(ndarray[np.int64_t, ndim=2] a, axis, **kwargs):
    cdef long *arr = <long*> a.data
    cdef int i, j
    cdef int nc = a.shape[1]
    cdef int nr = a.shape[0]
    cdef ndarray[np.int64_t] amin

    if axis == 0:
        amin = np.empty(nc, dtype=np.int64)
        amin.fill(MAX_INT)
        for i in range(nc):
            for j in range(nr):
                if arr[i * nr + j] < amin[i]:
                    amin[i] = arr[i * nr + j]
    else:
        amin = np.empty(nr, dtype=np.int64)
        amin.fill(MAX_INT)
        for i in range(nc):
            for j in range(nr):
                if arr[i * nr + j] < amin[j]:
                    amin[j] = arr[i * nr + j]
    return amin


def max_bool(ndarray[np.uint8_t, ndim=2, cast=True] a, axis, **kwargs):
    cdef unsigned char *arr = <unsigned char*> a.data
    cdef int i, j
    cdef int nc = a.shape[1]
    cdef int nr = a.shape[0]
    cdef ndarray[np.uint8_t] amax
    if axis == 0:
        amax = np.zeros(nc, dtype=np.uint8)
        for i in range(nc):
            for j in range(nr):
                if arr[i * nr + j] == 1:
                    amax[i] = 1
                    break
    else:
        amax = np.zeros(nr, dtype=np.uint8)
        for i in range(nc):
            for j in range(nr):
                if arr[i * nr + j] == 1:
                    amax[j] = 1
                    break
    return amax.astype(np.int64)

def min_bool(ndarray[np.uint8_t, ndim=2, cast=True] a, axis, **kwargs):
    cdef unsigned char *arr = <unsigned char*> a.data
    cdef int i, j
    cdef int nc = a.shape[1]
    cdef int nr = a.shape[0]
    cdef ndarray[np.uint8_t] amin

    if axis == 0:
        amin = np.ones(nc, dtype=np.uint8)
        for i in range(nc):
            for j in range(nr):
                if arr[i * nr + j] == 0:
                    amin[i] = 0
                    break
    else:
        amin = np.ones(nr, dtype=np.uint8)
        for i in range(nc):
            for j in range(nr):
                if arr[i * nr + j] == 0:
                    amin[j] = 0
                    break
    return amin.astype(np.int64)

def max_float(ndarray[np.float64_t, ndim=2] a, axis, hasnans):
    cdef double *arr = <double*> a.data
    cdef int i, j, k
    cdef int nc = a.shape[1]
    cdef int nr = a.shape[0]
    cdef ndarray[np.float64_t] amax

    if axis == 0:
        amax = np.empty(nc, dtype=np.float64)
        amax.fill(nan)
        for i in range(nc):
            if hasnans[i] is None or hasnans[i] == True:
                k = 0
                while isnan(arr[i * nr + k]) and k < nr - 1:
                    k += 1
                amax[i] = arr[i * nr + k]
                for j in range(k, nr):
                    if not isnan(arr[i * nr + j]):
                        if arr[i * nr + j] > amax[i]:
                            amax[i] = arr[i * nr + j]
            else:
                amax[i] = arr[i * nr]
                for j in range(nr):
                    if arr[i * nr + j] > amax[i]:
                        amax[i] = arr[i * nr + j]
    else:
        amax = np.empty(nr, dtype=np.float64)
        amax.fill(nan)
        if hasnans.sum() > 0:
            for i in range(nr):
                k = 0
                while isnan(arr[k * nr + i]) and k < nc - 1:
                    k += 1
                amax[i] = arr[k * nr + i]
                for j in range(k, nc):
                    if not isnan(arr[j * nr + i]):
                        if arr[j * nr + i] > amax[i]:
                            amax[i] = arr[j * nr + i]
        else:
            for i in range(nr):
                for j in range(nc):
                    if arr[j * nr + i] > amax[i]:
                        amax[i] = arr[j * nr + i]
    return amax

def min_float(ndarray[np.float64_t, ndim=2] a, axis, hasnans):
    cdef double *arr = <double*> a.data
    cdef int i, j, k
    cdef int nc = a.shape[1]
    cdef int nr = a.shape[0]
    cdef ndarray[np.float64_t] amin

    if axis == 0:
        amin = np.empty(nc, dtype=np.float64)
        amin.fill(nan)
        for i in range(nc):
            if hasnans[i] is None or hasnans[i] == True:
                k = 0
                while isnan(arr[i * nr + k]) and k < nr - 1:
                    k += 1
                amin[i] = arr[i * nr + k]
                for j in range(k, nr):
                    if not isnan(arr[i * nr + j]):
                        if arr[i * nr + j] < amin[i]:
                            amin[i] = arr[i * nr + j]
            else:
                amin[i] = arr[i * nr]
                for j in range(nr):
                    if arr[i * nr + j] < amin[i]:
                        amin[i] = arr[i * nr + j]
    else:
        amin = np.empty(nr, dtype=np.float64)
        amin.fill(nan)
        if hasnans.sum() > 0:
            for i in range(nr):
                k = 0
                while isnan(arr[k * nr + i]) and k < nc - 1:
                    k += 1
                amin[i] = arr[k * nr + i]
                for j in range(k, nc):
                    if not isnan(arr[j * nr + i]):
                        if arr[j * nr + i] < amin[i]:
                            amin[i] = arr[j * nr + i]
        else:
            for i in range(nr):
                for j in range(nc):
                    if arr[j * nr + i] < amin[i]:
                        amin[i] = arr[j * nr + i]
    return amin

def max_str(ndarray[object, ndim=2] a, axis, hasnans):
    cdef int i, j, k
    cdef int nc = a.shape[1]
    cdef int nr = a.shape[0]
    cdef ndarray[object] amax

    if axis == 0:
        amax = np.empty(nc, dtype='O')
        amax.fill(nan)
        for i in range(nc):
            if hasnans[i] is None or hasnans[i] == True:
                k = 0
                while a[k, i] is nan and k < nr:
                    k += 1
                amax[i] = a[k, i]
                for j in range(k, nr):
                    if a[j, i] is not nan:
                        if a[j, i] > amax[i]:
                            amax[i] = a[j, i]
            else:
                amax[i] = a[0, i]
                for j in range(nr):
                    if a[j, i] > amax[i]:
                        amax[i] = a[j, i]
    else:
        amax = np.empty(nr, dtype='O')
        amax.fill(nan)
        if hasnans.sum() > 0:
            for i in range(nr):
                k = 0
                while a[i, k] is nan and k < nc:
                    k += 1
                amax[i] = a[i, k]
                for j in range(k, nc):
                    if not a[i, j] is nan:
                        if a[i, j]  > amax[i]:
                            amax[i] = a[i, j] 
        else:
            for i in range(nr):
                for j in range(nc):
                    if a[i, j]  > amax[i]:
                        amax[i] = a[i, j]
    return amax

def min_str(ndarray[object, ndim=2] a, axis, hasnans):
    cdef int i, j, k
    cdef int nc = a.shape[1]
    cdef int nr = a.shape[0]
    cdef ndarray[object] amin

    if axis == 0:
        amin = np.empty(nc, dtype='O')
        amin.fill(nan)
        for i in range(nc):
            if hasnans[i] is None or hasnans[i] == True:
                k = 0
                while a[k, i] is nan and k < nr:
                    k += 1
                amin[i] = a[k, i]
                for j in range(k, nr):
                    if not a[j, i] is nan:
                        if a[j, i] < amin[i]:
                            amin[i] = a[j, i]
            else:
                amin[i] = a[0, i]
                for j in range(nr):
                    if a[j, i] < amin[i]:
                        amin[i] = a[j, i]
    else:
        amin = np.empty(nr, dtype='O')
        amin.fill(nan)
        if hasnans.sum() > 0:
            for i in range(nr):
                k = 0
                while a[i, k] is nan and k < nc:
                    k += 1
                amin[i] = a[i, k]
                for j in range(k, nc):
                    if not a[i, j] is nan:
                        if a[i, j] < amin[i]:
                            amin[i] = a[i, j]
        else:
            for i in range(nr):
                for j in range(nc):
                    if a[i, j] < amin[i]:
                        amin[i] = a[i, j]
    return amin

def mean_int(ndarray[np.int64_t, ndim=2] a, axis, **kwargs):
    cdef long *arr = <long*> a.data
    cdef int i, j
    cdef int nc = a.shape[1]
    cdef int nr = a.shape[0]
    cdef ndarray[np.int64_t] total

    if axis == 0:
        total = np.zeros(nc, dtype=np.int64)
        for i in range(nc):
            for j in range(nr):
                total[i] += arr[i * nr + j]
        return total / nr
    else:
        total = np.zeros(nr, dtype=np.int64)
        for i in range(nc):
            for j in range(nr):
                total[j] += arr[i * nr + j]
        return total / nc

def mean_bool(ndarray[np.uint8_t, ndim=2, cast=True] a, axis, **kwargs):
    cdef unsigned char *arr = <unsigned char*> a.data
    cdef int i, j
    cdef int nc = a.shape[1]
    cdef int nr = a.shape[0]
    cdef ndarray[np.int64_t] total

    if axis == 0:
        total = np.zeros(nc, dtype='int64')
        for i in range(nc):
            for j in range(nr):
                total[i] += arr[i * nr + j]
        return total / nr
    else:
        total = np.zeros(nr, dtype='int64')
        for i in range(nc):
            for j in range(nr):
                total[i] += arr[j * nr + i]
        return total / nc

def mean_float(ndarray[np.float64_t, ndim=2] a, axis, hasnans):
    cdef double *arr = <double*> a.data
    cdef int i, j
    cdef int nc = a.shape[1]
    cdef int nr = a.shape[0]
    cdef int ct = 0
    cdef ndarray[np.float64_t] total

    if axis == 0:
        total = np.zeros(nc, dtype=np.float64)
        for i in range(nc):
            if hasnans[i] is None or hasnans[i] == True:
                ct = 0
                for j in range(nr):
                    if not isnan(arr[i * nr + j]):
                        total[i] += arr[i * nr + j]
                        ct += 1
                if ct != 0:
                    total[i] = total[i] / ct
                else:
                    total[i] = nan
            else:
                for i in range(nc):
                    for j in range(nr):
                        total[i] += arr[i * nr + j]
                    total[i] = total[i] / nc
    else:
        total = np.zeros(nr, dtype=np.float64)
        for i in range(nr):
            ct = 0
            for j in range(nc):
                if not isnan(arr[j * nr + i]):
                    total[i] += arr[j * nr + i]
                    ct += 1
            if ct != 0:
                total[i] = total[i] / ct
            else:
                total[i] = nan
    return total

def median_int(ndarray[np.int64_t, ndim=2] a, axis, **kwargs):
    return np.median(a, axis=axis)

def median_bool(ndarray[np.uint8_t, cast=True, ndim=2] a, axis, **kwargs):
    return np.median(a, axis=axis)

def median_float(ndarray[np.float64_t, ndim=2] a, axis, hasnans):
    if axis == 0:
        if hasnans.any():
            return bn.nanmedian(a, axis=0)
        return np.median(a, axis=0)
    else:
        return bn.nanmedian(a, axis=1)

def var_float(ndarray[double, ndim=2] a, axis, int ddof, hasnans):

    cdef double *x = <double*> a.data
    cdef int i, j, i1
    cdef int ct = 0
    cdef int n = len(a)

    cdef int nc = a.shape[1]
    cdef int nr = a.shape[0]
    cdef ndarray[np.float64_t] total
    
    cdef double K = nan
    cdef double Ex = 0
    cdef double Ex2 = 0

    if axis == 0:
        total = np.zeros(nc, dtype=np.float64)
        for i in range(nc):
            i1 = 0
            K = x[i * nr + i1]
            while isnan(K):
                i1 += 1
                K = x[i * nr + i1]
            Ex = 0
            Ex2 = 0
            ct = 0
            for j in range(i1, nr):
                if isnan(x[i * nr + j]):
                    continue
                ct += 1
                Ex += x[i * nr + j] - K
                Ex2 += (x[i * nr + j] - K) * (x[i * nr + j] - K)
            if ct <= ddof:
                total[i] = nan
            else:
                total[i] = (Ex2 - (Ex * Ex) / ct) / (ct - ddof)
    else:
        total = np.zeros(nr, dtype=np.float64)
        for i in range(nr):
            i1 = 0
            K = x[i1 * nr + i]
            while isnan(K):
                i1 += 1
                K = x[i1 * nr + i]
            Ex = 0
            Ex2 = 0
            ct = 0
            for j in range(i1, nc):
                if isnan(x[j * nr + i]):
                    continue
                ct += 1
                Ex += x[j * nr + i] - K
                Ex2 += (x[j * nr + i] - K) * (x[j * nr + i] - K)
            if ct <= ddof:
                total[i] = nan
            else:
                total[i] = (Ex2 - (Ex * Ex) / ct) / (ct - ddof)

    return total

def var_int(ndarray[np.int64_t, ndim=2] a, axis, int ddof, hasnans):

    cdef long *x = <long*> a.data
    cdef int i, j
    cdef int nc = a.shape[1]
    cdef int nr = a.shape[0]
    cdef ndarray[np.float64_t] total
    
    cdef double K
    cdef double Ex = 0
    cdef double Ex2 = 0

    if axis == 0:
        total = np.zeros(nc, dtype=np.float64)
        for i in range(nc):
            if nr <= ddof:
                total[i] = nan
                continue
            K = x[i * nr]
            Ex = 0
            Ex2 = 0
            for j in range(nr):
                Ex += x[i * nr + j] - K
                Ex2 += (x[i * nr + j] - K) * (x[i * nr + j] - K)
            
            total[i] = (Ex2 - (Ex * Ex) / nr) / (nr - ddof)
    else:
        total = np.zeros(nr, dtype=np.float64)
        for i in range(nr):
            if nc <= ddof:
                total[i] = nan
                continue
            K = x[i]
            Ex = 0
            Ex2 = 0
            for j in range(nc):
                Ex += x[j * nr + i] - K
                Ex2 += (x[j * nr + i] - K) * (x[j * nr + i] - K)
            
            total[i] = (Ex2 - (Ex * Ex) / nc) / (nc - ddof)
    return total

def var_bool(ndarray[np.uint8_t, ndim=2, cast=True] a, axis, int ddof, hasnans):

    cdef unsigned char *x = <unsigned char *> a.data
    cdef int i, j
    cdef int nc = a.shape[1]
    cdef int nr = a.shape[0]
    cdef ndarray[np.float64_t] total
    
    cdef double K
    cdef double Ex = 0
    cdef double Ex2 = 0

    if axis == 0:
        total = np.zeros(nc, dtype=np.float64)
        for i in range(nc):
            if nr <= ddof:
                total[i] = nan
                continue
            K = x[i * nr]
            Ex = 0
            Ex2 = 0
            for j in range(nr):
                Ex += x[i * nr + j] - K
                Ex2 += (x[i * nr + j] - K) * (x[i * nr + j] - K)
            
            total[i] = (Ex2 - (Ex * Ex) / nr) / (nr - ddof)
    else:
        total = np.zeros(nr, dtype=np.float64)
        for i in range(nr):
            if nc <= ddof:
                total[i] = nan
                continue
            K = x[i]
            Ex = 0
            Ex2 = 0
            for j in range(nc):
                Ex += x[j * nr + i] - K
                Ex2 += (x[j * nr + i] - K) * (x[j * nr + i] - K)
            
            total[i] = (Ex2 - (Ex * Ex) / nc) / (nc - ddof)
    return total

def std_float(ndarray[np.float64_t, ndim=2] a, axis, int ddof, hasnans):
    return np.sqrt(var_float(a, axis, ddof, hasnans))

def std_int(ndarray[np.int64_t, ndim=2] a, axis, int ddof, hasnans):
    return np.sqrt(var_int(a, axis, ddof, hasnans))

def std_bool(ndarray[np.uint8_t, cast=True, ndim=2] a, axis, int ddof, hasnans):
    return np.sqrt(var_bool(a, axis, ddof, hasnans))

def any_int(ndarray[np.int64_t, ndim=2] a, axis, hasnans):
    cdef long *arr = <long*> a.data
    cdef int i, j
    cdef int nc = a.shape[1]
    cdef int nr = a.shape[0]
    cdef ndarray[np.uint8_t, cast=True] result

    if axis == 0:
        result = np.empty(nc, dtype='bool')
        for i in range(nc):
            result[i] = False
            for j in range(nr):
                if arr[i * nr + j] != 0:
                    result[i] = True
                    break
    else:
        result = np.empty(nr, dtype='bool')
        for i in range(nr):
            result[i] = False
            for j in range(nc):
                if arr[j * nr + i] != 0:
                    result[i] = True
                    break
    return result

def any_bool(ndarray[np.uint8_t, ndim=2, cast=True] a, axis, hasnans):
    cdef unsigned char *arr = <unsigned char*> a.data
    cdef int i, j
    cdef int nc = a.shape[1]
    cdef int nr = a.shape[0]
    cdef ndarray[np.uint8_t, cast=True] result

    if axis == 0:
        result = np.empty(nc, dtype='bool')
        for i in range(nc):
            result[i] = False
            for j in range(nr):
                if arr[i * nr + j] == True:
                    result[i] = True
                    break
    else:
        result = np.empty(nr, dtype='bool')
        for i in range(nr):
            result[i] = False
            for j in range(nc):
                if arr[j * nr + i] == True:
                    result[i] = True
                    break
    return result

def any_float(ndarray[np.float64_t, ndim=2] a, axis, hasnans):
    cdef double *arr = <double*> a.data
    cdef int i, j
    cdef int nc = a.shape[1]
    cdef int nr = a.shape[0]
    cdef ndarray[np.uint8_t, cast=True] result

    if axis == 0:
        result = np.empty(nc, dtype='bool')
        for i in range(nc):
            result[i] = False
            for j in range(nr):
                if arr[i * nr + j] != 0 and not isnan(arr[i * nr +j]):
                    result[i] = True
                    break
    else:
        result = np.empty(nr, dtype='bool')
        for i in range(nr):
            result[i] = False
            for j in range(nc):
                if arr[j * nr + i] != 0 and not isnan(arr[j * nr +i]):
                    result[i] = True
                    break
    return result

def any_str(ndarray[object, ndim=2] a, axis, hasnans):
    cdef int i, j
    cdef int nc = a.shape[1]
    cdef int nr = a.shape[0]
    cdef ndarray[np.uint8_t, cast=True] result

    if axis == 0:
        result = np.empty(nc, dtype='bool')
        for i in range(nc):
            result[i] = False
            for j in range(nr):
                if a[j, i] != '':
                    try:
                        if isnan(a[j, i]):
                            pass
                    except TypeError:
                        result[i] = True
                        break
    else:
        result = np.empty(nr, dtype='bool')
        for i in range(nr):
            result[i] = False
            for j in range(nc):
                if a[i, j] != '':
                    try:
                        if isnan(a[i, j]):
                            pass
                    except TypeError:
                        result[i] = True
                        break
    return result

def all_int(ndarray[np.int64_t, ndim=2] a, axis, hasnans):
    cdef long *arr = <long*> a.data
    cdef int i, j
    cdef int nc = a.shape[1]
    cdef int nr = a.shape[0]
    cdef ndarray[np.uint8_t, cast=True] result
    if axis == 0:
        result = np.empty(nc, dtype='bool')
        for i in range(nc):
            result[i] = True
            for j in range(nr):
                if arr[i * nr + j] == 0:
                    result[i] = False
                    break
    else:
        result = np.empty(nr, dtype='bool')
        for i in range(nr):
            result[i] = True
            for j in range(nc):
                if arr[j * nr + i] == 0:
                    result[i] = False
                    break
    return result

def all_bool(ndarray[np.uint8_t, ndim=2, cast=True] a, axis, hasnans):
    cdef unsigned char *arr = <unsigned char*> a.data
    cdef int i, j
    cdef int nc = a.shape[1]
    cdef int nr = a.shape[0]
    cdef ndarray[np.uint8_t, cast=True] result

    if axis == 0:
        result = np.empty(nc, dtype='bool')
        for i in range(nc):
            result[i] = True
            for j in range(nr):
                if arr[i * nr + j] == False:
                    result[i] = False
                    break
    else:
        result = np.empty(nr, dtype='bool')
        for i in range(nr):
            result[i] = True
            for j in range(nc):
                if arr[j * nr + i] == False:
                    result[i] = False
                    break
    return result

def all_float(ndarray[np.float64_t, ndim=2] a, axis, hasnans):
    cdef double *arr = <double*> a.data
    cdef int i, j
    cdef int nc = a.shape[1]
    cdef int nr = a.shape[0]
    cdef ndarray[np.uint8_t, cast=True] result

    if axis == 0:
        result = np.empty(nc, dtype='bool')
        for i in range(nc):
            result[i] = True
            for j in range(nr):
                if arr[i * nr + j] == 0 or isnan(arr[i * nr +j]):
                    result[i] = False
                    break
    else:
        result = np.empty(nr, dtype='bool')
        for i in range(nr):
            result[i] = True
            for j in range(nc):
                if arr[j * nr + i] == 0 or isnan(arr[j * nr +i]):
                    result[i] = False
                    break
    return result

def all_str(ndarray[object, ndim=2] a, axis, hasnans):
    cdef int i, j
    cdef int nc = a.shape[1]
    cdef int nr = a.shape[0]
    cdef ndarray[np.uint8_t, cast=True] result

    if axis == 0:
        result = np.empty(nc, dtype='bool')
        for i in range(nc):
            result[i] = True
            for j in range(nr):
                if a[j, i] != '':
                    try:
                        if a[j, i] is nan:
                            result[i] = False
                            break
                    except TypeError:
                        pass
                else:
                    result[i] = False
                    break
    else:
        result = np.empty(nr, dtype='bool')
        for i in range(nr):
            result[i] = True
            for j in range(nc):
                if a[i, j] != '':
                    try:
                        if a[i, j] is nan:
                            result[i] = False
                            break
                    except TypeError:
                        pass
                else:
                    result[i] = False
                    break
    return result

def argmax_int(ndarray[np.int64_t, ndim=2] a, axis, hasnans):
    cdef long *arr = <long*> a.data
    cdef int i, j
    cdef int nc = a.shape[1]
    cdef int nr = a.shape[0]
    cdef long amax
    cdef ndarray[np.int64_t] result

    if axis == 0:
        result = np.zeros(nc, dtype=np.int64)
        for i in range(nc):
            amax = arr[i * nr] 
            for j in range(nr):
                if arr[i * nr + j] > amax:
                    amax = arr[i * nr + j]
                    result[i] = j
    else:
        result = np.zeros(nr, dtype=np.int64)
        for i in range(nr):
            amax = arr[i] 
            for j in range(nc):
                if arr[j * nr + i] > amax:
                    amax = arr[j * nr + i]
                    result[i] = j
    return result

def argmin_int(ndarray[np.int64_t, ndim=2] a, axis, hasnans):
    cdef long *arr = <long*> a.data
    cdef int i, j
    cdef int nc = a.shape[1]
    cdef int nr = a.shape[0]
    cdef long amin
    cdef ndarray[np.int64_t] result

    if axis == 0:
        result = np.zeros(nc, dtype=np.int64)
        for i in range(nc):
            amin = arr[i * nr]
            for j in range(nr):
                if arr[i * nr + j] < amin:
                    amin = arr[i * nr + j]
                    result[i] = j
    else:
        result = np.zeros(nr, dtype=np.int64)
        for i in range(nr):
            amin = arr[i] 
            for j in range(nc):
                if arr[j * nr + i] < amin:
                    amin = arr[j * nr + i]
                    result[i] = j
    return result

def argmax_bool(ndarray[np.uint8_t, cast=True, ndim=2] a, axis, hasnans):
    cdef unsigned char *arr = <unsigned char*> a.data
    cdef int i, j
    cdef int nc = a.shape[1]
    cdef int nr = a.shape[0]
    cdef ndarray[np.int64_t] result

    if axis == 0:
        result = np.zeros(nc, dtype=np.int64)
        for i in range(nc):
            for j in range(nr):
                if arr[i * nr + j]  == True:
                    result[i] = j
                    break
    else:
        result = np.zeros(nr, dtype=np.int64)
        for i in range(nr):
            for j in range(nc):
                if arr[j * nr + i]  == True:
                    result[i] = j
                    break
    return result

def argmin_bool(ndarray[np.uint8_t, cast=True, ndim=2] a, axis, hasnans):
    cdef unsigned char *arr = <unsigned char*> a.data
    cdef int i, j
    cdef int nc = a.shape[1]
    cdef int nr = a.shape[0]
    cdef ndarray[np.int64_t] result

    if axis == 0:
        result = np.zeros(nc, dtype=np.int64)
        for i in range(nc):
            for j in range(nr):
                if arr[i * nr + j]  == False:
                    result[i] = j
                    break
    else:
        result = np.zeros(nr, dtype=np.int64)
        for i in range(nr):
            for j in range(nc):
                if arr[j * nr + i]  == False:
                    result[i] = j
                    break
    return result

def argmax_float(ndarray[np.float64_t, ndim=2] a, axis, hasnans):
    cdef double *arr = <double*> a.data
    cdef int i, j
    cdef int nc = a.shape[1]
    cdef int nr = a.shape[0]
    cdef long iloc = -1
    cdef double amax
    cdef ndarray[np.float64_t] result

    if axis == 0:
        result = np.empty(nc, dtype=np.float64)
        for i in range(nc):
            amax = MIN_FLOAT
            for j in range(nr):
                if arr[i * nr + j] > amax:
                    amax = arr[i * nr + j]
                    iloc = j
            if amax <= MIN_FLOAT + 1:
                result[i] = np.nan
            else:
                result[i] = iloc
    else:
        result = np.empty(nr, dtype=np.float64)
        for i in range(nr):
            amax = MIN_FLOAT
            for j in range(nc):
                if arr[j * nr + i] > amax:
                    amax = arr[j * nr + i]
                    iloc = j
            if amax <= MIN_FLOAT + 1:
                result[i] = np.nan
            else:
                result[i] = iloc
    return result

def argmin_float(ndarray[np.float64_t, ndim=2] a, axis, hasnans):
    cdef double *arr = <double*> a.data
    cdef int i, j
    cdef int nc = a.shape[1]
    cdef int nr = a.shape[0]
    cdef long iloc = -1
    cdef double amin
    cdef ndarray[np.float64_t] result

    if axis == 0:
        result = np.empty(nc, dtype=np.float64)
        for i in range(nc):
            amin = MAX_FLOAT
            for j in range(nr):
                if arr[i * nr + j] < amin:
                    amin = arr[i * nr + j]
                    iloc = j
            if amin >= MAX_FLOAT - 1:
                result[i] = np.nan
            else:
                result[i] = iloc
    else:
        result = np.empty(nr, dtype=np.float64)
        for i in range(nr):
            amin = MAX_FLOAT
            for j in range(nc):
                if arr[j * nr + i] < amin:
                    amin = arr[j * nr + i]
                    iloc = j
            if amin >= MAX_FLOAT - 1:
                result[i] = np.nan
            else:
                result[i] = iloc
    return result

def argmax_str(ndarray[object, ndim=2] a, axis, hasnans):
    cdef int i, j
    cdef int nc = a.shape[1]
    cdef int nr = a.shape[0]
    cdef long iloc = -1
    cdef str amax
    cdef ndarray[np.float64_t] result

    if axis == 0:
        result = np.empty(nc, dtype=np.float64)
        for i in range(nc):
            amax = MIN_CHAR
            for j in range(nr):
                try:
                    if a[j, i] > amax:
                        amax = a[j, i]
                        iloc = j
                except TypeError:
                    pass
            if amax == MIN_CHAR:
                result[i] = nan
            else:
                result[i] = iloc
    else:
        result = np.empty(nr, dtype=np.float64)
        for i in range(nr):
            amax = MIN_CHAR
            for j in range(nc):
                try:
                    if a[i, j] > amax:
                        amax = a[i, j]
                        iloc = j
                except TypeError:
                    pass
            if amax == MIN_CHAR:
                result[i] = nan
            else:
                result[i] = iloc
    return result

def argmin_str(ndarray[object, ndim=2] a, axis, hasnans):
    cdef int i, j
    cdef int nc = a.shape[1]
    cdef int nr = a.shape[0]
    cdef long iloc = -1
    cdef str amin
    cdef ndarray[np.float64_t] result

    if axis == 0:
        result = np.empty(nc, dtype=np.float64)
        for i in range(nc):
            amin = MAX_CHAR
            for j in range(nr):
                try:
                    if a[j, i] < amin:
                        amin = a[j, i]
                        iloc = j
                except TypeError:
                    pass
            if amin == MAX_CHAR:
                result[i] = nan
            else:
                result[i] = iloc
    else:
        result = np.empty(nr, dtype=np.float64)
        for i in range(nr):
            amin = MAX_CHAR
            for j in range(nc):
                try:
                    if a[i, j] < amin:
                        amin = a[i, j]
                        iloc = j
                except TypeError:
                    pass
            if amin == MAX_CHAR:
                result[i] = nan
            else:
                result[i] = iloc
    return result

def count_int(ndarray[np.int64_t, ndim=2] a, axis, hasnans):
    if axis == 0:
        result = np.empty(a.shape[0], dtype=np.int64)
        result.fill(a.shape[0])
    else:
        result = np.empty(a.shape[0], dtype=np.int64)
        result.fill(a.shape[1])
    return result

def count_bool(ndarray[np.uint8_t, cast=True, ndim=2] a, axis, hasnans):
    if axis == 0:
        result = np.empty(a.shape[0], dtype=np.int64)
        result.fill(a.shape[0])
    else:
        result = np.empty(a.shape[0], dtype=np.int64)
        result.fill(a.shape[1])
    return result

def count_float(ndarray[np.float64_t, ndim=2] a, axis, hasnans):
    cdef double *arr = <double*> a.data
    cdef int i, j
    cdef int nc = a.shape[1]
    cdef int nr = a.shape[0]
    cdef long ct
    cdef ndarray[np.int64_t] result

    if axis == 0:
        result = np.zeros(nc, dtype=np.int64)
        for i in range(nc):
            ct = 0
            for j in range(nr):
                if not isnan(arr[i * nr + j]):
                    ct += 1
            result[i] = ct
    else:
        result = np.zeros(nr, dtype=np.int64)
        for i in range(nr):
            ct = 0
            for j in range(nc):
                if not isnan(arr[j * nr + i]):
                    ct += 1
            result[i] = ct
    return result
            
def count_str(ndarray[object, ndim=2] a, axis, hasnans):
    cdef int i, j
    cdef int nc = a.shape[1]
    cdef int nr = a.shape[0]
    cdef long ct
    cdef ndarray[np.int64_t] result

    if axis == 0:
        result = np.zeros(nc, dtype=np.int64)
        for i in range(nc):
            ct = 0
            for j in range(nr):
                if not a[j, i] is nan:
                    ct += 1
            result[i] = ct
    else:
        result = np.zeros(nr, dtype=np.int64)
        for i in range(nr):
            ct = 0
            for j in range(nc):
                if not a[i, j] is nan:
                    ct += 1
            result[i] = ct
    return result

def clip_str(ndarray[object, ndim=2] a, str lower, str upper, hasnans):
    cdef int i, j
    cdef int nc = a.shape[1]
    cdef int nr = a.shape[0]
    cdef ndarray[object, ndim=2] b = np.empty((nr, nc), dtype='O')
    if hasnans == True or hasnans is None:
        for i in range(nc):
            for j in range(nr):
                if a[j, i] is nan:
                    b[j, i] = nan
                else:
                    if a[j, i] < lower:
                        b[j, i] = lower
                    elif a[j, i] > upper:
                        b[j, i] = upper
                    else:
                        b[j, i] = a[j, i]
        return b
    return a.clip(lower, upper)

def cummax_float(ndarray[np.float64_t, ndim=2] a, axis, hasnans):
    cdef np.float64_t *arr = <np.float64_t*> a.data
    cdef int i, j, k = 0
    cdef int nc = a.shape[1]
    cdef int nr = a.shape[0]
    cdef np.float64_t amax
    cdef ndarray[np.float64_t, ndim=2] b

    if axis == 0:
        b = np.empty((nr, nc), dtype=np.float64, order='F')
        for i in range(nc):
            k = 0
            amax = arr[i * nr + k]
            b[k, i] = amax
            while isnan(amax) and k < nr - 1:
                k += 1
                amax = arr[i * nr + k]
                b[k, i] = nan
            for j in range(k, nr):
                if arr[i * nr + j] > amax:
                    amax = arr[i * nr + j]
                b[j, i] = amax
    else:
        b = np.empty((nr, nc), dtype=np.float64, order='F')
        for i in range(nr):
            k = 0
            amax = arr[k * nr + i]
            b[i, k] = amax
            while isnan(amax) and k < nc - 1:
                k += 1
                amax = arr[k * nr + i]
                b[i, k] = nan
            for j in range(k, nc):
                if arr[j * nr + i] > amax:
                    amax = arr[j * nr + i]
                b[i, j] = amax
    return b

def cummin_float(ndarray[np.float64_t, ndim=2] a, axis, hasnans):
    cdef np.float64_t *arr = <np.float64_t*> a.data
    cdef int i, j, k = 0
    cdef int nc = a.shape[1]
    cdef int nr = a.shape[0]
    cdef np.float64_t amin
    cdef ndarray[np.float64_t, ndim=2] b

    if axis == 0:
        b = np.empty((nr, nc), dtype=np.float64, order='F')
        for i in range(nc):
            k = 0
            amin = arr[i * nr + k]
            b[k, i] = amin
            while isnan(amin) and k < nr - 1:
                k += 1
                amin = arr[i * nr + k]
                b[k, i] = nan
            for j in range(k, nr):
                if arr[i * nr + j] < amin:
                    amin = arr[i * nr + j]
                b[j, i] = amin
    else:
        b = np.empty((nr, nc), dtype=np.float64, order='F')
        for i in range(nr):
            k = 0
            amin = arr[k * nr + i]
            b[i, k] = amin
            while isnan(amin) and k < nc - 1:
                k += 1
                amin = arr[k * nr + i]
                b[i, k] = nan
            for j in range(k, nc):
                if arr[j * nr + i] < amin:
                    amin = arr[j * nr + i]
                b[i, j] = amin
    return b

def cummax_int(ndarray[np.int64_t, ndim=2] a, axis, hasnans):
    cdef np.int64_t *arr = <np.int64_t*> a.data
    cdef int i, j
    cdef int nc = a.shape[1]
    cdef int nr = a.shape[0]
    cdef np.int64_t amax
    cdef ndarray[np.int64_t, ndim=2] b = np.empty((nr, nc), dtype=np.int64)

    if axis == 0:
        b = np.empty((nr, nc), dtype=np.int64)
        for i in range(nc):
            amax = arr[i * nr]
            for j in range(nr):
                if arr[i * nr + j] > amax:
                    amax = arr[i * nr + j]
                b[j, i] = amax
    else:
        b = np.empty((nr, nc), dtype=np.int64)
        for i in range(nr):
            amax = arr[i]
            for j in range(nc):
                if arr[j * nr + i] > amax:
                    amax = arr[j * nr + i]
                b[i, j] = amax
    return b

def cummin_int(ndarray[np.int64_t, ndim=2] a, axis, hasnans):
    cdef np.int64_t *arr = <np.int64_t*> a.data
    cdef int i, j
    cdef int nc = a.shape[1]
    cdef int nr = a.shape[0]
    cdef np.int64_t amin
    cdef ndarray[np.int64_t, ndim=2] b = np.empty((nr, nc), dtype=np.int64)

    if axis == 0:
        b = np.empty((nr, nc), dtype=np.int64)
        for i in range(nc):
            amin = arr[i * nr]
            for j in range(nr):
                if arr[i * nr + j] < amin:
                    amin = arr[i * nr + j]
                b[j, i] = amin
    else:
        b = np.empty((nr, nc), dtype=np.int64)
        for i in range(nr):
            amin = arr[i]
            for j in range(nc):
                if arr[j * nr + i] < amin:
                    amin = arr[j * nr + i]
                b[i, j] = amin
    return b

def cummax_bool(ndarray[np.uint8_t, cast=True, ndim=2] a, axis, hasnans):
    cdef unsigned char *arr = <unsigned char*> a.data
    cdef int i, j
    cdef int nc = a.shape[1]
    cdef int nr = a.shape[0]
    cdef int amax = 0
    cdef ndarray[np.uint8_t, ndim=2, cast=True] b

    if axis == 0:
        for i in range(nc):
            b = np.empty((nr, nc), dtype='bool')
            amax = False
            for j in range(nr):
                if amax == True:
                    b[j, i] = True
                elif arr[i * nr + j] == True:
                    amax = True
                    b[j, i] = True
                else:
                    b[j, i] = False
    else:
        for i in range(nr):
            b = np.empty((nr, nc), dtype='bool')
            amax = False
            for j in range(nc):
                if amax == True:
                    b[i, j] = True
                elif arr[j * nr + i] == True:
                    amax = True
                    b[i, j] = True
                else:
                    b[i, j] = False
    return b

def cummin_bool(ndarray[np.uint8_t, cast=True, ndim=2] a, axis, hasnans):
    cdef unsigned char *arr = <unsigned char*> a.data
    cdef int i, j
    cdef int nc = a.shape[1]
    cdef int nr = a.shape[0]
    cdef int amin = 0
    cdef ndarray[np.uint8_t, ndim=2, cast=True] b

    if axis == 0:
        for i in range(nc):
            b = np.empty((nr, nc), dtype='bool')
            amin = True
            for j in range(nr):
                if amin == False:
                    b[j, i] = False
                elif arr[i * nr + j] == False:
                    amin = False
                    b[j, i] = False
                else:
                    b[j, i] = True
    else:
        for i in range(nr):
            b = np.empty((nr, nc), dtype='bool')
            amin = True
            for j in range(nc):
                if amin == False:
                    b[i, j] = False
                elif arr[j * nr + i] == False:
                    amin = False
                    b[i, j] = False
                else:
                    b[i, j] = True
    return b

def cummax_str(ndarray[object, ndim=2] a, axis, hasnans):
    cdef int i, j, ct
    cdef int nc = a.shape[1]
    cdef int nr = a.shape[0]
    cdef str amax
    cdef ndarray[object, ndim=2] b

    if axis == 0:
        b = np.empty((nr, nc), dtype='O')
        for i in range(nc):
            amax = ''
            ct = 0
            for j in range(nr):
                if a[j, i] is nan:
                    if ct == 0:
                        b[j, i] = nan
                    else:
                        b[j, i] = amax 
                else:
                    ct = 1
                    if a[j, i] > amax:
                        amax = a[j, i]
                    b[j, i] = amax
    else:
        b = np.empty((nr, nc), dtype='O')
        for i in range(nr):
            amax = ''
            ct = 0
            for j in range(nc):
                if a[i, j] is nan:
                    if ct == 0:
                        b[i, j] = nan
                    else:
                        b[i, j] = amax 
                else:
                    ct = 1
                    if a[i, j] > amax:
                        amax = a[i, j]
                    b[i, j] = amax
    return b

def cummin_str(ndarray[object, ndim=2] a, axis, hasnans):
    cdef int i, j
    cdef int nc = a.shape[1]
    cdef int nr = a.shape[0]
    cdef str amin
    cdef ndarray[object, ndim=2] b

    if axis == 0:
        b = np.empty((nr, nc), dtype='O')
        for i in range(nc):
            amin = MAX_CHAR
            for j in range(nr):
                if a[j, i] is nan:
                    if amin == MAX_CHAR:
                        b[j, i] = nan
                    else:
                        b[j, i] = amin 
                else:
                    if a[j, i] < amin:
                        amin = a[j, i]
                    b[j, i] = amin
    else:
        b = np.empty((nr, nc), dtype='O')
        for i in range(nr):
            amin = MAX_CHAR
            for j in range(nc):
                if a[i, j] is nan:
                    if amin == MAX_CHAR:
                        b[i, j] = nan
                    else:
                        b[i, j] = amin 
                else:
                    if a[i, j] < amin:
                        amin = a[i, j]
                    b[i, j] = amin
    return b

def cumsum_float(ndarray[np.float64_t, ndim=2] a, axis, hasnans):
    cdef np.float64_t *arr = <np.float64_t*> a.data
    cdef int i, j, k
    cdef int nc = a.shape[1]
    cdef int nr = a.shape[0]
    cdef ndarray[np.float64_t, ndim=2] total = np.empty((nr, nc), dtype=np.float64)
    total.fill(nan)
    cdef double cur_total

    if axis == 0:
        for i in range(nc):
            k = 0
            cur_total = arr[i * nr + k]
            while isnan(cur_total) and k < nr - 1:
                k += 1
                cur_total = arr[i * nr + k]
            total[k, i] = cur_total
            for j in range(k + 1, nr):
                if not isnan(arr[i * nr + j]):
                    cur_total += arr[i * nr + j]
                total[j, i] = cur_total
    else:
        for i in range(nr):
            k = 0
            cur_total = arr[k * nr + i]
            while isnan(cur_total) and k < nc - 1:
                k += 1
                cur_total = arr[k * nr + i]
            total[i, k] = cur_total
            for j in range(k + 1, nc):
                if not isnan(arr[j * nr + i]):
                    cur_total += arr[j * nr + i]
                total[i, j] = cur_total
    return total

def cumsum_int(ndarray[np.int64_t, ndim=2] a, axis, hasnans):
    cdef np.int64_t *arr = <np.int64_t*> a.data
    cdef int i, j
    cdef int nc = a.shape[1]
    cdef int nr = a.shape[0]
    cdef ndarray[np.int64_t, ndim=2] total = np.empty((nr, nc), dtype=np.int64)
    cdef np.int64_t cur_total

    if axis == 0:
        for i in range(nc):
            cur_total = 0
            for j in range(nr):
                cur_total += arr[i * nr + j]
                total[j, i] = cur_total
    else:
        for i in range(nr):
            cur_total = 0
            for j in range(nc):
                cur_total += arr[j * nr + i]
                total[i, j] = cur_total
    return total

def cumsum_bool(ndarray[np.int8_t, ndim=2, cast=True] a, axis, hasnans):
    cdef np.int8_t *arr = <np.int8_t*> a.data
    cdef int i, j
    cdef int nc = a.shape[1]
    cdef int nr = a.shape[0]
    cdef ndarray[np.int64_t, ndim=2] total = np.empty((nr, nc), dtype=np.int64)
    cdef np.int64_t cur_total

    if axis == 0:
        for i in range(nc):
            cur_total = 0
            for j in range(nr):
                cur_total += arr[i * nr + j]
                total[j, i] = cur_total
    else:
        for i in range(nr):
            cur_total = 0
            for j in range(nc):
                cur_total += arr[j * nr + i]
                total[i, j] = cur_total
    return total

def cumsum_str(ndarray[object, ndim=2] a, axis, hasnans):
    cdef int i, j, k
    cdef int nc = a.shape[1]
    cdef int nr = a.shape[0]
    cdef str cur_total
    cdef ndarray[object, ndim=2] total = np.empty((nr, nc), dtype='O')
    total.fill(nan)
    
    if axis == 0:
        for i in range(nc):
            k = 0
            cur_total = a[k, i]
            while cur_total is nan and k < nr - 1:
                k += 1
                cur_total = a[k, i]
            total[k, i] = cur_total
            for j in range(k + 1, nr):
                try:
                    cur_total += a[j, i]
                except TypeError:
                    pass
                total[j, i] = cur_total
    else:
        for i in range(nr):
            k = 0
            cur_total = a[i, k]
            while cur_total is nan and k < nc - 1:
                k += 1
                cur_total = a[i, k]
            total[k, i] = cur_total
            for j in range(k + 1, nc):
                try:
                    cur_total += a[i, j]
                except TypeError:
                    pass
                total[i, j] = cur_total
    return total

def isna_str(ndarray[object, ndim=2] a, ndarray[np.uint8_t, cast=True] hasnans):
    cdef int i, j
    cdef int nc = a.shape[1]
    cdef int nr = a.shape[0]
    cdef ndarray[np.int8_t, cast=True, ndim=2] b = np.zeros((nr, nc), dtype='bool')
    for i in range(nc):
        if hasnans[i] is False:
            continue
        for j in range(nr):
            b[j, i] = a[j, i] is nan
    return b

def isna_float(ndarray[np.float64_t, ndim=2] a, ndarray[np.uint8_t, cast=True] hasnans):
    cdef np.float64_t *arr = <np.float64_t*> a.data
    cdef int i, j
    cdef int nc = a.shape[1]
    cdef int nr = a.shape[0]
    cdef ndarray[np.int8_t, cast=True, ndim=2] b = np.zeros((nr, nc), dtype=bool)
    for i in range(nc):
        if hasnans[i] is False:
            continue
        for j in range(nr):
            b[j, i] = isnan(arr[i * nr + j])
    return b

def cov(ndarray[double] x, ndarray[double] y):
    cdef int i
    cdef int n = len(x)
    cdef int ct = 0
    if (n < 2):
        return np.nan
    cdef double kx, ky
    cdef double Ex = 0
    cdef double Ey = 0
    cdef double Exy = 0
    cdef double x_diff, y_diff
    cdef double cov

    kx = x[0]
    ky = y[0]
    Ex = Ey = Exy = 0
    for i in range(n):
        if isnan(x[i]) or isnan(y[i]):
            continue
        ct += 1
        x_diff = x[i] - kx
        y_diff = y[i] - ky
        Ex += x_diff
        Ey += y_diff
        Exy += x_diff * y_diff
    if ct == 0:
        return nan
    cov = (Exy - Ex * Ey / ct) / (ct - 1)
    return cov

def corr(ndarray[double] x, ndarray[double] y):
    cdef int i
    cdef int n = len(x)
    cdef int ct = 0
    if (n < 2):
        return np.nan
    cdef double kx, ky
    cdef double Ex = 0
    cdef double Ey = 0
    cdef double Exy = 0
    
    cdef double Ex2 = 0
    cdef double Ey2 = 0
    cdef double x_diff, y_diff
    cdef double cov, stdx, stdy
    kx = x[0]
    ky = y[0]
    Ex = Ey = Exy = 0
    for i in range(n):
        if isnan(x[i]) or isnan(y[i]):
            continue
        ct += 1
        x_diff = x[i] - kx
        y_diff = y[i] - ky
        Ex += x_diff
        Ey += y_diff
        Exy += x_diff * y_diff
        
        Ex2 += x_diff ** 2
        Ey2 += y_diff ** 2
    if ct < 2:
        return np.nan
    cov = (Exy - Ex * Ey / ct) / (ct - 1)
    stdx = (Ex2 - (Ex * Ex) / ct)/(ct - 1)
    stdy = (Ey2 - (Ey * Ey) / ct)/(ct - 1)
    return cov / (np.sqrt(stdx) * np.sqrt(stdy))
