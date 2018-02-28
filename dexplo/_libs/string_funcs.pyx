

# check for valid regex
# look up the except at end def double evaluate(self, double x) except *:
import numpy as np
from numpy import nan
from numpy cimport ndarray
import re
from typing import Pattern

cimport cython
cimport numpy as np

DTYPE = np.int8
ctypedef np.int8_t DTYPE_t


def capitalize(ndarray[object] arr):
    cdef int i
    cdef int n = len(arr)
    cdef ndarray[object] result = np.empty(n, dtype='object')
    for i in range(n):
        if arr[i] is not None:
            result[i] = arr[i].capitalize()
        else:
            result[i] = None
    return result

def capitalize_2d(ndarray[object, ndim=2] arr):
    cdef int i, j
    cdef int nr = arr.shape[0]
    cdef int nc = arr.shape[1]
    cdef ndarray[object, ndim=2] result = np.empty((nr, nc), dtype='object')
    for i in range(nr):
        for j in range(nc):
            if arr[i, j] is not None:
                result[i, j] = arr[i, j].capitalize()
            else:
                arr[i, j] = None
    return result

def center(ndarray[object] arr, int width, str fill_character=' '):
    cdef int i
    cdef int n = len(arr)
    cdef ndarray[object] result = np.empty(n, dtype='object')
    for i in range(n):
        if arr[i] is not None:
            result[i] = arr[i].center(width, fill_character)
        else:
            result[i] = None
    return result

def center_2d(ndarray[object, ndim=2] arr, int width, str fill_character=' '):
    cdef int i, j
    cdef int nr = len(arr)
    cdef int nc = arr.shape[1]
    cdef ndarray[object, ndim=2] result = np.empty((nr, nc), dtype='object')
    for i in range(nr):
        for j in range(nc):
            if arr[i, j] is not None:
                result[i, j] = arr[i, j].center(width, fill_character)
            else:
                result[i, j] = None
    return result

def contains(ndarray[object] arr, pat, case=True, flags=0, na=nan, regex=True):
    cdef int i
    cdef int n = len(arr)
    cdef ndarray[np.uint8_t, cast=True] result = np.empty(n, dtype='bool')

    if regex:
        if isinstance(pat, Pattern):
            pattern = pat
        else:
            if not case:
                flags = flags | re.IGNORECASE
            pattern = re.compile(pat, flags=flags)

        for i in range(n):
            if arr[i] is not None:
                result[i] = bool(pattern.search(arr[i]))
            else:
                result[i] = False
    else:
        if case:
            for i in range(n):
                if arr[i] is not None:
                    result[i] = pat in arr[i]
                else:
                    result[i] = False
        else:
            pat = pat.lower()
            for i in range(n):
                if arr[i] is not None:
                    result[i] = pat in arr[i].lower()
                else:
                    result[i] = False
    return result

def contains_2d(ndarray[object, ndim=2] arr, pat, case=True, flags=0, na=nan, regex=True):
    cdef int i, j
    cdef int nr = len(arr)
    cdef int nc = arr.shape[1]
    cdef ndarray[np.uint8_t, cast=True, ndim=2] result = np.empty((nr, nc), dtype='bool')

    if regex:
        if isinstance(pat, Pattern):
            pattern = pat
        else:
            if not case:
                flags = flags | re.IGNORECASE
            pattern = re.compile(pat, flags=flags)

        for i in range(nr):
            for j in range(nc):
                if arr[i, j] is not None:
                    result[i, j] = bool(pattern.search(arr[i, j]))
                else:
                    result[i, j] = False
    else:
        if case:
            for i in range(nr):
                for j in range(nc):
                    if arr[i, j] is not None:
                        result[i, j] = pat in arr[i, j]
                    else:
                        result[i, j] = False
        else:
            pat = pat.lower()
            for i in range(nr):
                for j in range(nc):
                    if arr[i, j] is not None:
                        result[i, j] = pat in arr[i, j].lower()
                    else:
                        result[i, j] = False
    return result

def count(ndarray[object] arr, pat, case=True, flags=0, na=nan, regex=True):
    cdef int i
    cdef int n = len(arr)
    cdef ndarray[np.float64_t] result = np.empty(n, dtype=np.float64)

    if regex:
        if isinstance(pat, Pattern):
            pattern = pat
        else:
            if not case:
                flags = flags | re.IGNORECASE
            pattern = re.compile(pat, flags=flags)
        for i in range(n):
            if arr[i] is not None:
                result[i] = len(pattern.findall(arr[i]))
            else:
                result[i] = na
        return result
    else:
        for i in range(n):
            if arr[i] is not None:
                result[i] = arr[i].count(pat)
            else:
                result[i] = na
        return result

def count_2d(ndarray[object, ndim=2] arr, pat, case=True, flags=0, na=nan, regex=True):
    cdef int i, j
    cdef int nr = len(arr)
    cdef int nc = arr.shape[1]
    cdef ndarray[np.float64_t, ndim=2] result = np.empty((nr, nc), dtype=np.float64)

    if regex:
        if isinstance(pat, Pattern):
            pattern = pat
        else:
            if not case:
                flags = flags | re.IGNORECASE
            pattern = re.compile(pat, flags=flags)

        for i in range(nr):
            for j in range(nc):
                if arr[i, j] is not None:
                    result[i, j] = len(pattern.findall(arr[i, j]))
                else:
                    result[i, j] = na
        return result
    else:
        for i in range(nr):
            for j in range(nc):
                if arr[i, j] is not None:
                    result[i, j] = arr[i, j].count(pat)
                else:
                    result[i, j] = na
        return result

def decode(ndarray[object] arr, str encoding, str errors='strict'):
    cdef int i
    cdef int n = len(arr)
    cdef ndarray[object] result = np.empty(n, dtype='object')
    for i in range(n):
        try:
            result[i] = bytes.decode(arr[i], encoding, errors)
        except TypeError:
            result[i] = nan
    return result

def encode(ndarray[object] arr, str encoding, str errors='strict'):
    cdef int i
    cdef int n = len(arr)
    cdef ndarray[object] result = np.empty(n, dtype='object')
    for i in range(n):
        result[i] = arr[i].encode(encoding, errors)
    return result

def endswith(ndarray[object] arr, str pat):
    cdef int i
    cdef int n = len(arr)
    cdef ndarray[np.uint8_t, cast=True] result = np.empty(n, dtype='bool')
    for i in range(n):
        if arr[i] is not None:
            result[i] = arr[i].endswith(pat)
        else:
            result[i] = False
    return result

def endswith_2d(ndarray[object, ndim=2] arr, str pat):
    cdef int i, j
    cdef int nr = len(arr)
    cdef int nc = arr.shape[1]
    cdef ndarray[np.uint8_t, cast=True, ndim=2] result = np.empty((nr, nc), dtype='bool')
    for i in range(nr):
        for j in range(nc):
            if arr[i, j] is not None:
                result[i, j] = arr[i, j].endswith(pat)
            else:
                result[i, j] = False
    return result

def find(ndarray[object] arr, str sub, start, end):
    cdef int i
    cdef np.uint8_t hasnans = False
    cdef int n = len(arr)
    cdef ndarray[np.float64_t] result = np.empty(n, dtype='float64')
    for i in range(n):
        if arr[i] is not None:
            result[i] = arr[i].find(sub, start, end)
        else:
            hasnans = True
            result[i] = nan

    if not hasnans:
        return result.astype('int64')
    return result

def find_2d(ndarray[object, ndim=2] arr, str sub, start, end):
    cdef int i, j
    cdef np.uint8_t hasnans = False
    cdef int nr = len(arr)
    cdef int nc = arr.shape[1]
    cdef ndarray[np.float64_t, ndim=2] result = np.empty((nr, nc), dtype='float64')
    for i in range(nr):
        for j in range(nc):
            if arr[i, j] is not None:
                result[i, j] = arr[i, j].find(sub, start, end)
            else:
                hasnans = True
                result[i, j] = nan

    if not hasnans:
        return result.astype('int64')
    return result

def get(ndarray[object] arr, int idx):
    cdef int i
    cdef int n = len(arr)
    cdef ndarray[object] result = np.empty(n, dtype='object')
    for i in range(n):
        if arr[i] is not None:
            try:
                result[i] = arr[i][idx]
            except IndexError:
                result[i] = None
        else:
            result[i] = None
    return result

def get_2d(ndarray[object, ndim=2] arr, int idx):
    cdef int i, j
    cdef int nr = len(arr)
    cdef int nc = arr.shape[1]
    cdef ndarray[object, ndim=2] result = np.empty((nr, nc), dtype='object')
    for i in range(nr):
        for j in range(nc):
            if arr[i, j] is not None:
                try:
                    result[i, j] = arr[i, j][idx]
                except IndexError:
                    result[i, j] = None
            else:
                result[i, j] = None
    return result

def get_dummies(ndarray[object] arr, sep='|'):
    cdef int i, arr_num, ct = 0
    cdef int n = len(arr)
    cdef dict new_cols = {}
    cdef list new_arrs = []
    cdef ndarray[object] col_names

    for i in range(n):
        if arr[i] is not None:
            for val in arr[i].split(sep):
                arr_num = new_cols.get(val, -1)
                if arr_num == -1:
                    new_arrs.append(np.zeros(n, dtype='int64'))
                    new_arrs[ct][i] = 1
                    new_cols[val] = ct
                    ct += 1
                else:
                    new_arrs[arr_num][i] = 1

    col_names = np.empty(len(new_arrs), dtype='O')
    for k, v in new_cols.items():
        col_names[v] = k
    return np.column_stack(new_arrs), col_names

def lower(ndarray[object] arr):
    cdef int i
    cdef int n = len(arr)
    cdef ndarray[object] result = np.empty(n, dtype='object')
    for i in range(n):
        result[i] = arr[i].lower()
    return result

def title(ndarray[object] arr):
    cdef int i
    cdef int n = len(arr)
    cdef ndarray[object] result = np.empty(n, dtype='object')
    for i in range(n):
        result[i] = arr[i].title()
    return result

def upper(ndarray[object] arr):
    cdef int i
    cdef int n = len(arr)
    cdef ndarray[object] result = np.empty(n, dtype='object')
    for i in range(n):
        result[i] = arr[i].upper()
    return result
