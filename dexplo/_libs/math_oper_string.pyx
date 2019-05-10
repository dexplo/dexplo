#cython: boundscheck=False
#cython: wraparound=False
from collections import defaultdict
import numpy as np
cimport numpy as np
from numpy cimport ndarray

# arithmetic and comparison operations for strings

try:
    import bottleneck as bn
except ImportError:
    import numpy as bn

MAX_CHAR = chr(1_000_000)
MIN_CHAR = chr(0)

# when creating a completely new string, should we recalculate the mappings?
# or by OK if multiple codes map to the same string
def str__add__(dict str_reverse_map, str other):
    cdef int n
    cdef dict new_str_map = {}
    cdef dict new_str_reverse_map = {}
    cdef list new_list

    for loc, list_strings in str_reverse_map.items():
        new_list = []
        new_str_reverse_map[loc] = new_list
        for val in list_strings:
            if val is None:
                new_list.append(None)
            else:
                new_list.append(val + other)
        new_str_map[loc] = dict(zip(new_list, range(len(new_list))))

    return new_str_map, new_str_reverse_map

def str__radd__(dict str_reverse_map, str other):
    cdef int n
    cdef dict new_str_map = {}
    cdef dict new_str_reverse_map = {}
    cdef list new_list

    for loc, list_strings in str_reverse_map.items():
        new_list = []
        new_str_reverse_map[loc] = new_list
        for val in list_strings:
            if val is None:
                new_list.append(None)
            else:
                new_list.append(other + val)
        new_str_map[loc] = dict(zip(new_list, range(len(new_list))))

    return new_str_map, new_str_reverse_map

def str__add__arr(ndarray[np.uint32_t] arr1, ndarray[np.uint32_t] arr2,
                  list srm1, list srm2):
    cdef Py_ssize_t i, j
    cdef int nr = len(arr1), code, r=len(srm1), c=len(srm2), count=0, err=len(arr1)
    cdef dict str_map = {}
    cdef new_str_reverse_map = []
    cdef new_str_map = {}
    cdef ndarray[np.uint32_t] arr = np.empty(nr, 'uint32', 'F')
    cdef ndarray[object] str_combos

    if r * c < nr / 10:
        str_combos = np.empty(r * c,  'O', 'F')
        for i, val1 in enumerate(srm1):
            for j, val2 in enumerate(srm2):
                if val1 is not None and val2 is not None:
                    str_combos[i * r + j] = val1 + val2

        for i in range(nr):
            val = str_combos[arr1[i] * r + arr2[j]]
            code = new_str_map.get(val, -1)
            if code == -1:
                new_str_reverse_map.append(val)
                arr[i] = len(new_str_map)
                new_str_map[val] = len(new_str_map)
            else:
                arr[i] = code
    else:
        for i in range(nr):
            try:
                val = srm1[arr1[i]] + srm2[arr2[i]]
            except:
                val = None

            code = new_str_map.get(val, -1)
            if code == -1:
                new_str_reverse_map.append(val)
                arr[i] = len(new_str_map)
                new_str_map[val] = len(new_str_map)
            else:
                arr[i] = code

    return arr, new_str_map, new_str_reverse_map

def str__radd__arr(ndarray[np.uint32_t] arr1, ndarray[np.uint32_t] arr2,
                   list srm1, list srm2):
    return str__add__arr(arr2, arr1,  srm2, srm1)

def str__mul__(dict str_reverse_map, int other):
    cdef int n
    cdef dict new_str_map = {}
    cdef dict new_str_reverse_map = {}
    cdef list new_list

    for loc, list_strings in str_reverse_map.items():
        new_list = []
        new_str_reverse_map[loc] = new_list
        for val in list_strings:
            if val is None:
                new_list.append(None)
            else:
                new_list.append(val * other)
        new_str_map[loc] = dict(zip(new_list, range(len(new_list))))

    return new_str_map, new_str_reverse_map

def str__mul__arr(ndarray[object] arr, ndarray[object] arr2):
    cdef int i, j
    cdef int nr = arr.shape[0]
    cdef ndarray[object] final = np.empty(nr, dtype='O')
    for i in range(nr):
        if arr[i] is None or arr2[i] is None:
            final[i] = None
        else:
            final[i] = arr[i] * arr2[i]
    return final

def str__lt__(dict str_reverse_map, ndarray[np.uint32_t, ndim=2] arr_codes, str other):
    cdef int nr = len(arr_codes), nc=arr_codes.shape[1], i
    cdef ndarray[np.uint8_t, ndim=2, cast=True] final = np.empty((nr, nc), dtype='bool')
    cdef list list_strings, trues

    for loc, list_strings in str_reverse_map.items():
        trues = []
        for i, val in enumerate(list_strings):
            if val is not None and val < other:
                trues.append(i)
        final[:, loc] = np.isin(arr_codes[:, loc], trues)

    return final

def str__lt__arr(dict srm1, dict srm2, ndarray[np.uint32_t, ndim=2] arr1,
                 ndarray[np.uint32_t, ndim=2] arr2):
    cdef Py_ssize_t i, j
    cdef int nr = len(arr1)
    cdef ndarray[np.uint8_t, cast=True] final = np.empty(nr, dtype='bool')

    for i in range(nr):
        try:
            final[i] = srm1[arr1[i]] < srm2[arr2[i]]
        except:
            final[i] = False
    return final

def str__le__(dict str_reverse_map, ndarray[np.uint32_t, ndim=2] arr_codes, str other):
    cdef int nr = len(arr_codes), nc=arr_codes.shape[1], i
    cdef ndarray[np.uint8_t, ndim=2, cast=True] final = np.empty((nr, nc), dtype='bool')
    cdef list list_strings, trues

    for loc, list_strings in str_reverse_map.items():
        trues = []
        for i, val in enumerate(list_strings):
            if val is not None and val <= other:
                trues.append(i)
        final[:, loc] = np.isin(arr_codes[:, loc], trues)

    return final

def str__le__arr(ndarray[object] arr, ndarray[object] arr2):
    cdef int i, j
    cdef int nr = arr.shape[0]
    cdef ndarray[np.uint8_t, cast=True] final = np.empty(nr, dtype='bool')

    for i in range(nr):
        try:
            final[i] = arr[i] <= arr2[i]
        except:
            final[i] = False
    return final

def str__gt__(dict str_reverse_map, ndarray[np.uint32_t, ndim=2] arr_codes, str other):
    cdef int nr = len(arr_codes), nc=arr_codes.shape[1], i
    cdef ndarray[np.uint8_t, ndim=2, cast=True] final = np.empty((nr, nc), dtype='bool')
    cdef list list_strings, trues

    for loc, list_strings in str_reverse_map.items():
        trues = []
        for i, val in enumerate(list_strings):
            if val is not None and val > other:
                trues.append(i)
        final[:, loc] = np.isin(arr_codes[:, loc], trues)

    return final

def str__gt__arr(ndarray[object] arr, ndarray[object] arr2):
    cdef int i, j
    cdef int nr = arr.shape[0]
    cdef ndarray[np.uint8_t, cast=True] final = np.empty(nr, dtype='bool')

    for i in range(nr):
        try:
            final[i] = arr[i] > arr2[i]
        except:
            final[i] = False
    return final

def str__ge__(dict str_reverse_map, ndarray[np.uint32_t, ndim=2] arr_codes, str other):
    cdef int nr = len(arr_codes), nc=arr_codes.shape[1], i
    cdef ndarray[np.uint8_t, ndim=2, cast=True] final = np.empty((nr, nc), dtype='bool')
    cdef list list_strings, trues

    for loc, list_strings in str_reverse_map.items():
        trues = []
        for i, val in enumerate(list_strings):
            if val is not None and val >= other:
                trues.append(i)
        final[:, loc] = np.isin(arr_codes[:, loc], trues)

    return final

def str__ge__arr(ndarray[object] arr, ndarray[object] arr2):
    cdef int i, j
    cdef int nr = arr.shape[0]
    cdef ndarray[np.uint8_t, cast=True] final = np.empty(nr, dtype='bool')

    for i in range(nr):
        try:
            final[i] = arr[i] >= arr2[i]
        except:
            final[i] = False
    return final

def str__eq__(dict str_reverse_map, ndarray[np.uint32_t, ndim=2] arr_codes, str other):
    cdef int nr = len(arr_codes), nc=arr_codes.shape[1], i
    cdef ndarray[np.uint8_t, ndim=2, cast=True] final = np.empty((nr, nc), dtype='bool')
    cdef list list_strings, trues

    for loc, list_strings in str_reverse_map.items():
        trues = []
        for i, val in enumerate(list_strings):
            if val is not None and val == other:
                trues.append(i)
        final[:, loc] = np.isin(arr_codes[:, loc], trues)

    return final

def str__eq__arr(ndarray[object] arr, ndarray[object] arr2):
    cdef int i, j
    cdef int nr = arr.shape[0]
    cdef ndarray[np.uint8_t, cast=True] final = np.empty(nr, dtype='bool')

    for i in range(nr):
        try:
            final[i] = arr[i] == arr2[i]
        except:
            final[i] = False
    return final

def str__ne__(dict str_reverse_map, ndarray[np.uint32_t, ndim=2] arr_codes, str other):
    cdef int nr = len(arr_codes), nc=arr_codes.shape[1], i
    cdef ndarray[np.uint8_t, ndim=2, cast=True] final = np.empty((nr, nc), dtype='bool')
    cdef list list_strings, trues

    for loc, list_strings in str_reverse_map.items():
        trues = []
        for i, val in enumerate(list_strings):
            if val is not None and val != other:
                trues.append(i)
        final[:, loc] = np.isin(arr_codes[:, loc], trues)

    return final

def str__ne__arr(ndarray[object] arr, ndarray[object] arr2):
    cdef int i, j
    cdef int nr = arr.shape[0]
    cdef ndarray[np.uint8_t, cast=True] final = np.empty(nr, dtype='bool')

    for i in range(nr):
        try:
            final[i] = arr[i] != arr2[i]
        except:
            final[i] = False
    return final
