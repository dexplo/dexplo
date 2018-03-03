

# check for valid regex
# look up the except at end def double evaluate(self, double x) except *:
import numpy as np
from numpy import nan
from numpy cimport ndarray
import re
from typing import Pattern

cimport cython
cimport numpy as np


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

def center(ndarray[object] arr, int width, str fillchar=' '):
    cdef int i
    cdef int n = len(arr)
    cdef ndarray[object] result = np.empty(n, dtype='object')
    for i in range(n):
        if arr[i] is not None:
            result[i] = arr[i].center(width, fillchar)
        else:
            result[i] = None
    return result

def center_2d(ndarray[object, ndim=2] arr, int width, str fillchar=' '):
    cdef int i, j
    cdef int nr = len(arr)
    cdef int nc = arr.shape[1]
    cdef ndarray[object, ndim=2] result = np.empty((nr, nc), dtype='object')
    for i in range(nr):
        for j in range(nc):
            if arr[i, j] is not None:
                result[i, j] = arr[i, j].center(width, fillchar)
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

def findall(ndarray[object, ndim=2] arr, pat, pos, case=True, flags=0, int count=0):
    cdef int i, j, k, arr_num, ct = 0
    cdef int nr = arr.shape[0]
    cdef int nc = arr.shape[1]
    cdef list data = []
    cdef list result
    cdef ndarray[object] cur_arr
    cdef ndarray[object, ndim=2] final_arr
    cdef int amax = 0
    cdef int total_max = 0
    cdef list maxes = [0]

    if isinstance(pat, Pattern):
        pattern = pat
    else:
        if not case:
            flags = flags | re.IGNORECASE
        pattern = re.compile(pat, flags=flags)

    for j in range(nc):
        cur_arr = np.empty(nr, 'O')
        amax = 0
        for i in range(nr):
            if arr[i, j] is not None:
                result = pattern.findall(arr[i, j], pos)
                cur_arr[i] = result
                if len(result) > amax:
                    amax = len(result)
            else:
                cur_arr[i] = None
        total_max += amax
        data.append(cur_arr)
        maxes.append(amax)

    final_arr = np.empty((nr, total_max + count), 'O')
    for j in range(nc):
        cur_arr = data[j]
        for i in range(nr):
            if cur_arr[i] is not None:
                for k, val in enumerate(cur_arr[i]):
                    final_arr[i, k + maxes[j] + count] = val

    col_names = []
    for i, val in enumerate(maxes[1:]):
        for k in range(val):
            col_names.append(str(k))

    return final_arr, np.array(col_names, 'O'), maxes[1:]

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

def get_dummies(ndarray[object, ndim=2] arr, sep, int count):
    cdef int i, j
    cdef int nr = arr.shape[0]
    cdef int nc = arr.shape[1]
    cdef ndarray[np.int64_t] group = np.empty(nr, dtype=np.int64)
    cdef ndarray[np.int64_t, ndim=2] result
    cdef dict d = {}
    cdef list col_dicts = []
    cdef list col_groups = []
    cdef list group_len = []

    for j in range(nc):
        d = {}
        for i in range(nr):
            if arr[i, j] is not None:
                group[i] = d.get(arr[i, j], -1)
                if group[i] == -1:
                    group[i] = count
                    d[arr[i, j]] = count
                    count += 1

        col_dicts.append(d)
        col_groups.append(group)

    result = np.zeros((nr, count), dtype='int64', order='F')

    if sep is None:
        for j in range(nc):
            d = col_dicts[j]
            group = col_groups[j]
            group_len.append(len(d))
            for i in range(nr):
                if arr[i, j] is not None:
                    result[i, group[i]] = 1
    else:
        raise NotImplementedError('no sep yet')

    col_names = []
    for d in col_dicts:
        col_names.extend(list(d))

    return result, np.array(col_names, dtype='O'), group_len

def isalnum(ndarray[object] arr):
    cdef int i
    cdef int n = len(arr)
    cdef ndarray[np.uint8_t, cast=True] result = np.empty(n, dtype='bool')
    for i in range(n):
        if arr[i] is not None:
            result[i] = arr[i].isalnum()
        else:
            result[i] = False
    return result

def isalnum_2d(ndarray[object, ndim=2] arr):
    cdef int i, j
    cdef int nr = len(arr)
    cdef int nc = arr.shape[1]
    cdef ndarray[np.uint8_t, cast=True, ndim=2] result = np.empty((nr, nc), dtype='bool')
    for i in range(nr):
        for j in range(nc):
            if arr[i, j] is not None:
                result[i, j] = arr[i, j].isalnum()
            else:
                result[i, j] = False
    return result

def isalpha(ndarray[object] arr):
    cdef int i
    cdef int n = len(arr)
    cdef ndarray[np.uint8_t, cast=True] result = np.empty(n, dtype='bool')
    for i in range(n):
        if arr[i] is not None:
            result[i] = arr[i].isalpha()
        else:
            result[i] = False
    return result

def isalpha_2d(ndarray[object, ndim=2] arr):
    cdef int i, j
    cdef int nr = len(arr)
    cdef int nc = arr.shape[1]
    cdef ndarray[np.uint8_t, cast=True, ndim=2] result = np.empty((nr, nc), dtype='bool')
    for i in range(nr):
        for j in range(nc):
            if arr[i, j] is not None:
                result[i, j] = arr[i, j].isalpha()
            else:
                result[i, j] = False
    return result

def isdecimal(ndarray[object] arr):
    cdef int i
    cdef int n = len(arr)
    cdef ndarray[np.uint8_t, cast=True] result = np.empty(n, dtype='bool')
    for i in range(n):
        if arr[i] is not None:
            result[i] = arr[i].isdecimal()
        else:
            result[i] = False
    return result

def isdecimal_2d(ndarray[object, ndim=2] arr):
    cdef int i, j
    cdef int nr = len(arr)
    cdef int nc = arr.shape[1]
    cdef ndarray[np.uint8_t, cast=True, ndim=2] result = np.empty((nr, nc), dtype='bool')
    for i in range(nr):
        for j in range(nc):
            if arr[i, j] is not None:
                result[i, j] = arr[i, j].isdecimal()
            else:
                result[i, j] = False
    return result

def isdigit(ndarray[object] arr):
    cdef int i
    cdef int n = len(arr)
    cdef ndarray[np.uint8_t, cast=True] result = np.empty(n, dtype='bool')
    for i in range(n):
        if arr[i] is not None:
            result[i] = arr[i].isdigit()
        else:
            result[i] = False
    return result

def isdigit_2d(ndarray[object, ndim=2] arr):
    cdef int i, j
    cdef int nr = len(arr)
    cdef int nc = arr.shape[1]
    cdef ndarray[np.uint8_t, cast=True, ndim=2] result = np.empty((nr, nc), dtype='bool')
    for i in range(nr):
        for j in range(nc):
            if arr[i, j] is not None:
                result[i, j] = arr[i, j].isdigit()
            else:
                result[i, j] = False
    return result

def islower(ndarray[object] arr):
    cdef int i
    cdef int n = len(arr)
    cdef ndarray[np.uint8_t, cast=True] result = np.empty(n, dtype='bool')
    for i in range(n):
        if arr[i] is not None:
            result[i] = arr[i].islower()
        else:
            result[i] = False
    return result

def islower_2d(ndarray[object, ndim=2] arr):
    cdef int i, j
    cdef int nr = len(arr)
    cdef int nc = arr.shape[1]
    cdef ndarray[np.uint8_t, cast=True, ndim=2] result = np.empty((nr, nc), dtype='bool')
    for i in range(nr):
        for j in range(nc):
            if arr[i, j] is not None:
                result[i, j] = arr[i, j].islower()
            else:
                result[i, j] = False
    return result

def isnumeric(ndarray[object] arr):
    cdef int i
    cdef int n = len(arr)
    cdef ndarray[np.uint8_t, cast=True] result = np.empty(n, dtype='bool')
    for i in range(n):
        if arr[i] is not None:
            result[i] = arr[i].isnumeric()
        else:
            result[i] = False
    return result

def isnumeric_2d(ndarray[object, ndim=2] arr):
    cdef int i, j
    cdef int nr = len(arr)
    cdef int nc = arr.shape[1]
    cdef ndarray[np.uint8_t, cast=True, ndim=2] result = np.empty((nr, nc), dtype='bool')
    for i in range(nr):
        for j in range(nc):
            if arr[i, j] is not None:
                result[i, j] = arr[i, j].isnumeric()
            else:
                result[i, j] = False
    return result

def isspace(ndarray[object] arr):
    cdef int i
    cdef int n = len(arr)
    cdef ndarray[np.uint8_t, cast=True] result = np.empty(n, dtype='bool')
    for i in range(n):
        if arr[i] is not None:
            result[i] = arr[i].isspace()
        else:
            result[i] = False
    return result

def isspace_2d(ndarray[object, ndim=2] arr):
    cdef int i, j
    cdef int nr = len(arr)
    cdef int nc = arr.shape[1]
    cdef ndarray[np.uint8_t, cast=True, ndim=2] result = np.empty((nr, nc), dtype='bool')
    for i in range(nr):
        for j in range(nc):
            if arr[i, j] is not None:
                result[i, j] = arr[i, j].isspace()
            else:
                result[i, j] = False
    return result

def istitle(ndarray[object] arr):
    cdef int i
    cdef int n = len(arr)
    cdef ndarray[np.uint8_t, cast=True] result = np.empty(n, dtype='bool')
    for i in range(n):
        if arr[i] is not None:
            result[i] = arr[i].istitle()
        else:
            result[i] = False
    return result

def istitle_2d(ndarray[object, ndim=2] arr):
    cdef int i, j
    cdef int nr = len(arr)
    cdef int nc = arr.shape[1]
    cdef ndarray[np.uint8_t, cast=True, ndim=2] result = np.empty((nr, nc), dtype='bool')
    for i in range(nr):
        for j in range(nc):
            if arr[i, j] is not None:
                result[i, j] = arr[i, j].istitle()
            else:
                result[i, j] = False
    return result

def isupper(ndarray[object] arr):
    cdef int i
    cdef int n = len(arr)
    cdef ndarray[np.uint8_t, cast=True] result = np.empty(n, dtype='bool')
    for i in range(n):
        if arr[i] is not None:
            result[i] = arr[i].isupper()
        else:
            result[i] = False
    return result

def isupper_2d(ndarray[object, ndim=2] arr):
    cdef int i, j
    cdef int nr = len(arr)
    cdef int nc = arr.shape[1]
    cdef ndarray[np.uint8_t, cast=True, ndim=2] result = np.empty((nr, nc), dtype='bool')
    for i in range(nr):
        for j in range(nc):
            if arr[i, j] is not None:
                result[i, j] = arr[i, j].isupper()
            else:
                result[i, j] = False
    return result

def join(ndarray[object] arr, str sep):
    cdef int i
    cdef int n = len(arr)
    cdef ndarray[object] result = np.empty(n, dtype='O')
    for i in range(n):
        if arr[i] is not None:
            result[i] = sep.join(arr[i])
        else:
            result[i] = None
    return result

def join_2d(ndarray[object, ndim=2] arr, str sep):
    cdef int i, j
    cdef int nr = len(arr)
    cdef int nc = arr.shape[1]
    cdef ndarray[object, ndim=2] result = np.empty((nr, nc), dtype='O')
    for i in range(nr):
        for j in range(nc):
            if arr[i, j] is not None:
                result[i, j] = sep.join(arr[i, j])
            else:
                result[i, j] = None
    return result

def _len(ndarray[object] arr):
    cdef int i
    cdef int n = len(arr)
    cdef ndarray[np.int64_t] result = np.empty(n, dtype='int64')
    for i in range(n):
        if arr[i] is not None:
            result[i] = len(arr[i])
        else:
            result[i] = -1
    return result

def _len_2d(ndarray[object, ndim=2] arr):
    cdef int i, j
    cdef int nr = len(arr)
    cdef int nc = arr.shape[1]
    cdef ndarray[np.int64_t, ndim=2] result = np.empty((nr, nc), dtype='int64')
    for i in range(nr):
        for j in range(nc):
            if arr[i, j] is not None:
                result[i, j] = len(arr[i, j])
            else:
                result[i, j] = -1
    return result

def ljust(ndarray[object] arr, int width, str fillchar=' '):
    cdef int i
    cdef int n = len(arr)
    cdef ndarray[object] result = np.empty(n, dtype='object')
    for i in range(n):
        if arr[i] is not None:
            result[i] = arr[i].ljust(width, fillchar)
        else:
            result[i] = None
    return result

def ljust_2d(ndarray[object, ndim=2] arr, int width, str fillchar=' '):
    cdef int i, j
    cdef int nr = len(arr)
    cdef int nc = arr.shape[1]
    cdef ndarray[object, ndim=2] result = np.empty((nr, nc), dtype='object')
    for i in range(nr):
        for j in range(nc):
            if arr[i, j] is not None:
                result[i, j] = arr[i, j].ljust(width, fillchar)
            else:
                result[i, j] = None
    return result

def lower(ndarray[object] arr):
    cdef int i
    cdef int n = len(arr)
    cdef ndarray[object] result = np.empty(n, dtype='object')
    for i in range(n):
        if arr[i] is not None:
            result[i] = arr[i].lower()
        else:
            result[i] = None
    return result

def lower_2d(ndarray[object, ndim=2] arr):
    cdef int i, j
    cdef int nr = len(arr)
    cdef int nc = arr.shape[1]
    cdef ndarray[object, ndim=2] result = np.empty((nr, nc), dtype='object')
    for i in range(nr):
        for j in range(nc):
            if arr[i, j] is not None:
                result[i, j] = arr[i, j].lower()
            else:
                result[i, j] = None
    return result

def lstrip(ndarray[object] arr, to_strip):
    cdef int i
    cdef int n = len(arr)
    cdef ndarray[object] result = np.empty(n, dtype='object')
    for i in range(n):
        if arr[i] is not None:
            result[i] = arr[i].lstrip(to_strip)
        else:
            result[i] = None
    return result

def lstrip_2d(ndarray[object, ndim=2] arr, to_strip):
    cdef int i, j
    cdef int nr = len(arr)
    cdef int nc = arr.shape[1]
    cdef ndarray[object, ndim=2] result = np.empty((nr, nc), dtype='object')
    for i in range(nr):
        for j in range(nc):
            if arr[i, j] is not None:
                result[i, j] = arr[i, j].lstrip(to_strip)
            else:
                result[i, j] = None
    return result

def partition(ndarray[object, ndim=2] arr, str sep, int count):
    cdef int i, j
    cdef int nr = len(arr)
    cdef int nc = arr.shape[1]
    cdef ndarray[object, ndim=2] result = np.empty((nr, 3 * nc + count), dtype='object')

    for j in range(nc):
        for i in range(nr):
            if arr[i, j] is not None:
                for k, val in  enumerate(arr[i, j].partition(sep)):
                    result[i, j * 3 + k + count] = val
    return result, np.array(['head', 'sep', 'tail'] * nc, dtype='O'), [3] * nc

def repeat(ndarray[object] arr, int repeats):
    cdef int i
    cdef int n = len(arr)
    cdef ndarray[object] result = np.empty(n, dtype='object')
    for i in range(n):
        if arr[i] is not None:
            try:
                result[i] = arr[i] * repeats
            except IndexError:
                result[i] = None
        else:
            result[i] = None
    return result

def repeat_2d(ndarray[object, ndim=2] arr, int repeats):
    cdef int i, j
    cdef int nr = len(arr)
    cdef int nc = arr.shape[1]
    cdef ndarray[object, ndim=2] result = np.empty((nr, nc), dtype='object')
    for i in range(nr):
        for j in range(nc):
            if arr[i, j] is not None:
                try:
                    result[i, j] = arr[i, j] * repeats
                except IndexError:
                    result[i, j] = None
            else:
                result[i, j] = None
    return result

def replace(ndarray[object] arr, pat, str repl='', n=0, case=True, flags=0):
    cdef Py_ssize_t i
    cdef Py_ssize_t nr = len(arr)
    cdef ndarray[object] result = np.empty(nr, dtype='O')

    if isinstance(pat, Pattern):
        pattern = pat
    else:
        if not case:
            flags = flags | re.IGNORECASE
        pattern = re.compile(pat, flags=flags)

    for i in range(nr):
        if arr[i] is not None:
            result[i] = pattern.sub(repl, arr[i], n)
        else:
            result[i] = None
    return result

def replace_2d(ndarray[object, ndim=2] arr, pat, str repl='', n=0, case=True, flags=0):
    cdef Py_ssize_t i, j
    cdef Py_ssize_t nr = len(arr)
    cdef Py_ssize_t nc = arr.shape[1]
    cdef ndarray[object, ndim=2] result = np.empty((nr, nc), dtype='O')

    if isinstance(pat, Pattern):
        pattern = pat
    else:
        if not case:
            flags = flags | re.IGNORECASE
        pattern = re.compile(pat, flags=flags)

    for i in range(nr):
        for j in range(nc):
            if arr[i, j] is not None:
                result[i, j] = pattern.sub(repl, arr[i, j], n)
            else:
                result[i, j] = None

    return result

def rfind(ndarray[object] arr, str sub, start, end):
    cdef int i
    cdef np.uint8_t hasnans = False
    cdef int n = len(arr)
    cdef ndarray[np.float64_t] result = np.empty(n, dtype='float64')
    for i in range(n):
        if arr[i] is not None:
            result[i] = arr[i].rfind(sub, start, end)
        else:
            hasnans = True
            result[i] = nan

    if not hasnans:
        return result.astype('int64')
    return result

def rfind_2d(ndarray[object, ndim=2] arr, str sub, start, end):
    cdef int i, j
    cdef np.uint8_t hasnans = False
    cdef int nr = len(arr)
    cdef int nc = arr.shape[1]
    cdef ndarray[np.float64_t, ndim=2] result = np.empty((nr, nc), dtype='float64')
    for i in range(nr):
        for j in range(nc):
            if arr[i, j] is not None:
                result[i, j] = arr[i, j].rfind(sub, start, end)
            else:
                hasnans = True
                result[i, j] = nan

    if not hasnans:
        return result.astype('int64')
    return result

def rjust(ndarray[object] arr, int width, str fillchar=' '):
    cdef int i
    cdef int n = len(arr)
    cdef ndarray[object] result = np.empty(n, dtype='object')
    for i in range(n):
        if arr[i] is not None:
            result[i] = arr[i].rjust(width, fillchar)
        else:
            result[i] = None
    return result

def rjust_2d(ndarray[object, ndim=2] arr, int width, str fillchar=' '):
    cdef int i, j
    cdef int nr = len(arr)
    cdef int nc = arr.shape[1]
    cdef ndarray[object, ndim=2] result = np.empty((nr, nc), dtype='object')
    for i in range(nr):
        for j in range(nc):
            if arr[i, j] is not None:
                result[i, j] = arr[i, j].rjust(width, fillchar)
            else:
                result[i, j] = None
    return result

def rpartition(ndarray[object, ndim=2] arr, str sep, int count):
    cdef int i, j
    cdef int nr = len(arr)
    cdef int nc = arr.shape[1]
    cdef ndarray[object, ndim=2] result = np.empty((nr, 3 * nc + count), dtype='object')

    for j in range(nc):
        for i in range(nr):
            if arr[i, j] is not None:
                for k, val in  enumerate(arr[i, j].rpartition(sep)):
                    result[i, j * 3 + k + count] = val
    return result, np.array(['head', 'sep', 'tail'] * nc, dtype='O'), [3] * nc

# def rsplit(ndarray[object] arr, pat, n, case=True, flags=0):
#     cdef int i, j, arr_num, ct = 0
#     cdef int nr = len(arr)
#     cdef list new_arrs = []
#     cdef ndarray[object] col_names
#
#     if isinstance(pat, Pattern):
#         pattern = pat
#     else:
#         if not case:
#             flags = flags | re.IGNORECASE
#         pattern = re.compile(pat, flags=flags)
#
#     for i in range(nr):
#         if arr[i] is not None:
#             for j, val in enumerate((pattern.split(arr[i][::-1], n))[::-1]):
#                 try:
#                     new_arrs[j][i] = val[::-1]
#                 except IndexError:
#                     new_arrs.append(np.empty(nr, 'O'))
#                     new_arrs[j][i] = val[::-1]
#
#     col_names = np.array(['split_' + str(i) for i in range(len(new_arrs))], 'O')
#     return np.column_stack(new_arrs), col_names

def rsplit(ndarray[object, ndim=2] arr, pat, n, case=True, flags=0, int count=0):
    cdef int i, j, k, arr_num, ct = 0
    cdef int nr = arr.shape[0]
    cdef int nc = arr.shape[1]
    cdef list data = []
    cdef list result
    cdef ndarray[object] cur_arr
    cdef ndarray[object, ndim=2] final_arr
    cdef int amax = 0
    cdef int total_max = 0
    cdef list maxes = [0]

    if isinstance(pat, Pattern):
        pattern = pat
    else:
        if not case:
            flags = flags | re.IGNORECASE
        pattern = re.compile(pat, flags=flags)

    for j in range(nc):
        cur_arr = np.empty(nr, 'O')
        amax = 0
        for i in range(nr):
            if arr[i, j] is not None:
                result = pattern.split(arr[i, j][::-1], n)[::-1]
                cur_arr[i] = result
                if len(result) > amax:
                    amax = len(result)
            else:
                cur_arr[i] = None
        total_max += amax
        data.append(cur_arr)
        maxes.append(amax)

    final_arr = np.empty((nr, total_max + count), 'O')
    for j in range(nc):
        cur_arr = data[j]
        for i in range(nr):
            if cur_arr[i] is not None:
                for k, val in enumerate(cur_arr[i]):
                    final_arr[i, k + maxes[j] + count] = val[::-1]

    col_names = []
    for i, val in enumerate(maxes[1:]):
        for k in range(val):
            col_names.append(str(k))

    return final_arr, np.array(col_names, 'O'), maxes[1:]

def rstrip(ndarray[object] arr, to_strip):
    cdef int i
    cdef int n = len(arr)
    cdef ndarray[object] result = np.empty(n, dtype='object')
    for i in range(n):
        if arr[i] is not None:
            result[i] = arr[i].rstrip(to_strip)
        else:
            result[i] = None
    return result

def rstrip_2d(ndarray[object, ndim=2] arr, to_strip):
    cdef int i, j
    cdef int nr = len(arr)
    cdef int nc = arr.shape[1]
    cdef ndarray[object, ndim=2] result = np.empty((nr, nc), dtype='object')
    for i in range(nr):
        for j in range(nc):
            if arr[i, j] is not None:
                result[i, j] = arr[i, j].rstrip(to_strip)
            else:
                result[i, j] = None
    return result

def _slice(ndarray[object] arr, start, stop, step):
    cdef int i
    cdef int n = len(arr)
    cdef ndarray[object] result = np.empty(n, dtype='object')
    for i in range(n):
        if arr[i] is not None:
            result[i] = arr[i][start:stop:step]
        else:
            result[i] = None
    return result

def _slice_2d(ndarray[object, ndim=2] arr, start, stop, step):
    cdef int i, j
    cdef int nr = len(arr)
    cdef int nc = arr.shape[1]
    cdef ndarray[object, ndim=2] result = np.empty((nr, nc), dtype='object')
    for i in range(nr):
        for j in range(nc):
            if arr[i, j] is not None:
                result[i, j] = arr[i, j][start:stop:step]
            else:
                result[i, j] = None
    return result

def slice_replace(ndarray[object] arr, int start, stop, str repl):
    cdef int i
    cdef int n = len(arr)
    cdef ndarray[object] result = np.empty(n, dtype='object')

    if stop is None:
        for i in range(n):
            if arr[i] is not None:
                result[i] = arr[i][:start] + repl
            else:
                result[i] = None
    else:
        for i in range(n):
            if arr[i] is not None:
                result[i] = arr[i][:start] + repl + arr[i][stop:]
            else:
                result[i] = None
    return result

def slice_replace_2d(ndarray[object, ndim=2] arr, int start, stop, str repl):
    cdef int i, j
    cdef int nr = len(arr)
    cdef int nc = arr.shape[1]
    cdef ndarray[object, ndim=2] result = np.empty((nr, nc), dtype='object')
    cdef str s

    if stop is None:
        for i in range(nr):
            for j in range(nc):
                if arr[i, j] is not None:
                    result[i, j] = arr[i, j][:start] + repl
                else:
                    result[i, j] = None
    else:
        for i in range(nr):
            for j in range(nc):
                if arr[i, j] is not None:
                    result[i, j] = arr[i, j][:start] + repl + arr[i, j][stop:]
                else:
                    result[i, j] = None
    return result

def split(ndarray[object, ndim=2] arr, pat, n, case=True, flags=0, int count=0):
    cdef int i, j, k, arr_num, ct = 0
    cdef int nr = arr.shape[0]
    cdef int nc = arr.shape[1]
    cdef list data = []
    cdef list result
    cdef ndarray[object] cur_arr
    cdef ndarray[object, ndim=2] final_arr
    cdef int amax = 0
    cdef int total_max = 0
    cdef list maxes = [0]

    if isinstance(pat, Pattern):
        pattern = pat
    else:
        if not case:
            flags = flags | re.IGNORECASE
        pattern = re.compile(pat, flags=flags)

    for j in range(nc):
        cur_arr = np.empty(nr, 'O')
        amax = 0
        for i in range(nr):
            if arr[i, j] is not None:
                result = pattern.split(arr[i, j], n)
                cur_arr[i] = result
                if len(result) > amax:
                    amax = len(result)
            else:
                cur_arr[i] = None
        total_max += amax
        data.append(cur_arr)
        maxes.append(amax)

    final_arr = np.empty((nr, total_max + count), 'O')
    for j in range(nc):
        cur_arr = data[j]
        for i in range(nr):
            if cur_arr[i] is not None:
                for k, val in enumerate(cur_arr[i]):
                    final_arr[i, k + maxes[j] + count] = val

    col_names = []
    for i, val in enumerate(maxes[1:]):
        for k in range(val):
            col_names.append(str(k))

    return final_arr, np.array(col_names, 'O'), maxes[1:]

def startswith(ndarray[object] arr, str pat):
    cdef int i
    cdef int n = len(arr)
    cdef ndarray[np.uint8_t, cast=True] result = np.empty(n, dtype='bool')
    for i in range(n):
        if arr[i] is not None:
            result[i] = arr[i].startswith(pat)
        else:
            result[i] = False
    return result

def startswith_2d(ndarray[object, ndim=2] arr, str pat):
    cdef int i, j
    cdef int nr = len(arr)
    cdef int nc = arr.shape[1]
    cdef ndarray[np.uint8_t, cast=True, ndim=2] result = np.empty((nr, nc), dtype='bool')
    for i in range(nr):
        for j in range(nc):
            if arr[i, j] is not None:
                result[i, j] = arr[i, j].startswith(pat)
            else:
                result[i, j] = False
    return result

def swapcase(ndarray[object] arr):
    cdef int i
    cdef int n = len(arr)
    cdef ndarray[object] result = np.empty(n, dtype='object')
    for i in range(n):
        if arr[i] is not None:
            result[i] = arr[i].swapcase()
        else:
            result[i] = None
    return result

def swapcase_2d(ndarray[object, ndim=2] arr):
    cdef int i, j
    cdef int nr = len(arr)
    cdef int nc = arr.shape[1]
    cdef ndarray[object, ndim=2] result = np.empty((nr, nc), dtype='object')
    for i in range(nr):
        for j in range(nc):
            if arr[i, j] is not None:
                result[i, j] = arr[i, j].swapcase()
            else:
                result[i, j] = None
    return result

def title(ndarray[object] arr):
    cdef int i
    cdef int n = len(arr)
    cdef ndarray[object] result = np.empty(n, dtype='object')
    for i in range(n):
        if arr[i] is not None:
            result[i] = arr[i].title()
        else:
            result[i] = None
    return result

def title_2d(ndarray[object, ndim=2] arr):
    cdef int i, j
    cdef int nr = len(arr)
    cdef int nc = arr.shape[1]
    cdef ndarray[object, ndim=2] result = np.empty((nr, nc), dtype='object')
    for i in range(nr):
        for j in range(nc):
            if arr[i, j] is not None:
                result[i, j] = arr[i, j].title()
            else:
                result[i, j] = None
    return result

def upper(ndarray[object] arr):
    cdef int i
    cdef int n = len(arr)
    cdef ndarray[object] result = np.empty(n, dtype='object')
    for i in range(n):
        if arr[i] is not None:
            result[i] = arr[i].upper()
        else:
            result[i] = None
    return result

def upper_2d(ndarray[object, ndim=2] arr):
    cdef int i, j
    cdef int nr = len(arr)
    cdef int nc = arr.shape[1]
    cdef ndarray[object, ndim=2] result = np.empty((nr, nc), dtype='object')
    for i in range(nr):
        for j in range(nc):
            if arr[i, j] is not None:
                result[i, j] = arr[i, j].upper()
            else:
                result[i, j] = None
    return result

def strip(ndarray[object] arr, to_strip):
    cdef int i
    cdef int n = len(arr)
    cdef ndarray[object] result = np.empty(n, dtype='object')
    for i in range(n):
        if arr[i] is not None:
            result[i] = arr[i].strip(to_strip)
        else:
            result[i] = None
    return result

def strip_2d(ndarray[object, ndim=2] arr, to_strip):
    cdef int i, j
    cdef int nr = len(arr)
    cdef int nc = arr.shape[1]
    cdef ndarray[object, ndim=2] result = np.empty((nr, nc), dtype='object')
    for i in range(nr):
        for j in range(nc):
            if arr[i, j] is not None:
                result[i, j] = arr[i, j].strip(to_strip)
            else:
                result[i, j] = None
    return result

def translate(ndarray[object] arr, dict table):
    cdef int i
    cdef int n = len(arr)
    cdef ndarray[object] result = np.empty(n, dtype='object')
    for i in range(n):
        if arr[i] is not None:
            result[i] = arr[i].translate(table)
        else:
            result[i] = None
    return result

def translate_2d(ndarray[object, ndim=2] arr, dict table):
    cdef int i, j
    cdef int nr = len(arr)
    cdef int nc = arr.shape[1]
    cdef ndarray[object, ndim=2] result = np.empty((nr, nc), dtype='object')
    for i in range(nr):
        for j in range(nc):
            if arr[i, j] is not None:
                result[i, j] = arr[i, j].translate(table)
            else:
                result[i, j] = None
    return result

def zfill(ndarray[object] arr, int width):
    cdef int i
    cdef int n = len(arr)
    cdef ndarray[object] result = np.empty(n, dtype='object')
    for i in range(n):
        if arr[i] is not None:
            result[i] = arr[i].zfill(width)
        else:
            result[i] = None
    return result

def zfill_2d(ndarray[object, ndim=2] arr, int width):
    cdef int i, j
    cdef int nr = len(arr)
    cdef int nc = arr.shape[1]
    cdef ndarray[object, ndim=2] result = np.empty((nr, nc), dtype='object')
    for i in range(nr):
        for j in range(nc):
            if arr[i, j] is not None:
                result[i, j] = arr[i, j].zfill(width)
            else:
                result[i, j] = None
    return result

def wrap(ndarray[object] arr, t):
    cdef int i
    cdef int n = len(arr)
    cdef ndarray[object] result = np.empty(n, dtype='object')
    for i in range(n):
        if arr[i] is not None:
            result[i] = r'\n'.join(t.wrap(arr[i]))
        else:
            result[i] = None
    return result

def wrap_2d(ndarray[object, ndim=2] arr, t):
    cdef int i, j
    cdef int nr = len(arr)
    cdef int nc = arr.shape[1]
    cdef ndarray[object, ndim=2] result = np.empty((nr, nc), dtype='object')
    for i in range(nr):
        for j in range(nc):
            if arr[i, j] is not None:
                result[i, j] = r'\n'.join(t.wrap(arr[i, j]))
            else:
                result[i, j] = None
    return result
