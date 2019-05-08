#cython: boundscheck=False
#cython: wraparound=False
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
def add_str(dict str_reverse_map, str other):
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

def radd_str(dict str_reverse_map, str other):
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

def add_str_one(ndarray[np.uint32_t] arr1, ndarray[np.uint32_t] arr2,
                dict srm1, dict srm2):
    cdef Py_ssize_t i, j
    cdef int nr = len(arr1), code
    cdef dict str_map = {}
    cdef unicode val
    cdef new_str_reverse_map = []
    cdef new_str_map = {}
    cdef ndarray[np.uint32_t] arr = np.empty(nr, 'uint32', 'F')

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

def add_str_two(ndarray[np.uint32_t, ndim=2] arr1, ndarray[np.uint32_t, ndim=2] arr2,
                dict srm1, dict srm2):
    cdef Py_ssize_t i, j
    cdef int nr = len(arr1), nc = arr1.shape[1], code
    cdef dict str_map = {}
    cdef dict new_str_reverse_map = {}
    cdef dict new_str_map = {}, d
    cdef list list_map, cur_srm1, cur_srm2
    cdef ndarray[np.uint32_t, ndim=2] arr = np.empty((nr, nc), 'uint32', 'F')

    for j in range(nc):
        d = {}
        list_map = []
        new_str_reverse_map[j] = list_map
        new_str_map = d
        cur_srm1 = srm1[j]
        cur_srm2 = srm2[j]
        for i in range(nr):
            try:
                val = cur_srm1[arr1[i, j]] + cur_srm2[arr2[i, j]]
            except:
                val = None
            code = d.get(val, -1)
            if code == -1:
                list_map.append(val)
                arr[i, j] = len(d)
                d[val] = len(d)
            else:
                arr[i, j] = code

    return arr, new_str_map, new_str_reverse_map

def add_str_two_1row_right(ndarray[object, ndim=2] arr, ndarray[object] arr2):
    cdef int i, j
    cdef int nr = arr.shape[0]
    cdef int nc = arr.shape[1]
    cdef ndarray[object, ndim=2] final = np.empty((nr, nc), dtype='O')

    for i in range(nr):
        for j in range(nc):
            if arr[i, j] is None or arr2[j] is None:
                final[i, j] = None
            else:
                final[i, j] = arr[i, j] + arr2[j]
    return final

def add_str_two_1row_left(ndarray[object] arr, ndarray[object, ndim=2] arr2):
    cdef int i, j
    cdef int nr = arr2.shape[0]
    cdef int nc = arr2.shape[1]
    cdef ndarray[object, ndim=2] final = np.empty((nr, nc), dtype='O')

    for i in range(nr):
        for j in range(nc):
            if arr[j] is None or arr2[i, j] is None:
                final[i, j] = None
            else:
                final[i, j] = arr[j] + arr2[i, j]
    return final

def mul_str(dict str_reverse_map, int other):
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

def mul_str_one(ndarray[object] arr, ndarray[object] arr2):
    cdef int i, j
    cdef int nr = arr.shape[0]
    cdef ndarray[object] final = np.empty(nr, dtype='O')
    for i in range(nr):
        if arr[i] is None or arr2[i] is None:
            final[i] = None
        else:
            final[i] = arr[i] * arr2[i]
    return final

def mul_str_two(ndarray[object, ndim=2] arr, ndarray[object, ndim=2] arr2):
    cdef int i, j
    cdef int nr = arr.shape[0]
    cdef int nc = arr.shape[1]
    cdef ndarray[object, ndim=2] final = np.empty((nr, nc), dtype='O')
    for i in range(nr):
        for j in range(nc):
            if arr[i, j] is None or arr2[i, j] is None:
                final[i, j] = None
            else:
                final[i, j] = arr[i, j] * arr2[i, j]
    return final

def mul_str_two_1row_right(ndarray[object, ndim=2] arr, ndarray[object] arr2):
    cdef int i, j
    cdef int nr = arr.shape[0]
    cdef int nc = arr.shape[1]
    cdef ndarray[object, ndim=2] final = np.empty((nr, nc), dtype='O')

    for i in range(nr):
        for j in range(nc):
            if arr[i, j] is None or arr2[j] is None:
                final[i, j] = None
            else:
                final[i, j] = arr[i, j] * arr2[j]
    return final

def mul_str_two_1row_left(ndarray[object] arr, ndarray[object, ndim=2] arr2):
    cdef int i, j
    cdef int nr = arr2.shape[0]
    cdef int nc = arr2.shape[1]
    cdef ndarray[object, ndim=2] final = np.empty((nr, nc), dtype='O')

    for i in range(nr):
        for j in range(nc):
            if arr[j] is None or arr2[i, j] is None:
                final[i, j] = None
            else:
                final[i, j] = arr[j] * arr2[i, j]
    return final

def lt_str(dict str_reverse_map, str other, ndarray[np.uint32_t, ndim=2] arr_codes):
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

def lt_str_one(dict srm1, dict srm2, ndarray[np.uint32_t, ndim=2] arr1,
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

def lt_str_two(ndarray[object, ndim=2] arr, ndarray[object, ndim=2] arr2):
    cdef int i, j
    cdef int nr = arr.shape[0]
    cdef int nc = arr.shape[1]
    cdef ndarray[np.uint8_t, ndim=2, cast=True] final = np.empty((nr, nc), dtype='bool')

    for i in range(nr):
        for j in range(nc):
            try:
                final[i, j] = arr[i, j] < arr2[i, j]
            except:
                final[i, j] = False
    return final

def lt_str_two_1row_right(ndarray[object, ndim=2] arr, ndarray[object] arr2):
    cdef int i, j
    cdef int nr = arr.shape[0]
    cdef int nc = arr.shape[1]
    cdef ndarray[np.uint8_t, ndim=2, cast=True] final = np.empty((nr, nc), dtype='bool')

    for i in range(nr):
        for j in range(nc):
            try:
                final[i, j] = arr[i, j] < arr2[j]
            except:
                final[i, j] = False
    return final

def lt_str_two_1row_left(ndarray[object] arr, ndarray[object, ndim=2] arr2):
    cdef int i, j
    cdef int nr = arr2.shape[0]
    cdef int nc = arr2.shape[1]
    cdef ndarray[np.uint8_t, ndim=2, cast=True] final = np.empty((nr, nc), dtype='bool')

    for i in range(nr):
        for j in range(nc):
            try:
                final[i, j] = arr[j] < arr2[i, j]
            except:
                final[i, j] = False
    return final

def le_str(dict str_reverse_map, str other, ndarray[np.uint32_t, ndim=2] arr_codes):
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

def le_str_one(ndarray[object] arr, ndarray[object] arr2):
    cdef int i, j
    cdef int nr = arr.shape[0]
    cdef ndarray[np.uint8_t, cast=True] final = np.empty(nr, dtype='bool')

    for i in range(nr):
        try:
            final[i] = arr[i] <= arr2[i]
        except:
            final[i] = False
    return final

def le_str_two(ndarray[object, ndim=2] arr, ndarray[object, ndim=2] arr2):
    cdef int i, j
    cdef int nr = arr.shape[0]
    cdef int nc = arr.shape[1]
    cdef ndarray[np.uint8_t, ndim=2, cast=True] final = np.empty((nr, nc), dtype='bool')
    for i in range(nr):
        for j in range(nc):
            try:
                final[i, j] = arr[i, j] <= arr2[i, j]
            except:
                final[i, j] = False
    return final

def le_str_two_1row_right(ndarray[object, ndim=2] arr, ndarray[object] arr2):
    cdef int i, j
    cdef int nr = arr.shape[0]
    cdef int nc = arr.shape[1]
    cdef ndarray[np.uint8_t, ndim=2, cast=True] final = np.empty((nr, nc), dtype='bool')

    for i in range(nr):
        for j in range(nc):
            try:
                final[i, j] = arr[i, j] <= arr2[j]
            except:
                final[i, j] = False
    return final

def le_str_two_1row_left(ndarray[object] arr, ndarray[object, ndim=2] arr2):
    cdef int i, j
    cdef int nr = arr2.shape[0]
    cdef int nc = arr2.shape[1]
    cdef ndarray[np.uint8_t, ndim=2, cast=True] final = np.empty((nr, nc), dtype='bool')

    for i in range(nr):
        for j in range(nc):
            try:
                final[i, j] = arr[j] <= arr2[i, j]
            except:
                final[i, j] = False
    return final

def gt_str(dict str_reverse_map, str other, ndarray[np.uint32_t, ndim=2] arr_codes):
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

def gt_str_one(ndarray[object] arr, ndarray[object] arr2):
    cdef int i, j
    cdef int nr = arr.shape[0]
    cdef ndarray[np.uint8_t, cast=True] final = np.empty(nr, dtype='bool')

    for i in range(nr):
        try:
            final[i] = arr[i] > arr2[i]
        except:
            final[i] = False
    return final

def gt_str_two(ndarray[object, ndim=2] arr, ndarray[object, ndim=2] arr2):
    cdef int i, j
    cdef int nr = arr.shape[0]
    cdef int nc = arr.shape[1]
    cdef ndarray[np.uint8_t, ndim=2, cast=True] final = np.empty((nr, nc), dtype='bool')
    for i in range(nr):
        for j in range(nc):
            try:
                final[i, j] = arr[i, j] > arr2[i, j]
            except:
                final[i, j] = False
    return final

def gt_str_two_1row_right(ndarray[object, ndim=2] arr, ndarray[object] arr2):
    cdef int i, j
    cdef int nr = arr.shape[0]
    cdef int nc = arr.shape[1]
    cdef ndarray[np.uint8_t, ndim=2, cast=True] final = np.empty((nr, nc), dtype='bool')

    for i in range(nr):
        for j in range(nc):
            try:
                final[i, j] = arr[i, j] > arr2[j]
            except:
                final[i, j] = False
    return final

def gt_str_two_1row_left(ndarray[object] arr, ndarray[object, ndim=2] arr2):
    cdef int i, j
    cdef int nr = arr2.shape[0]
    cdef int nc = arr2.shape[1]
    cdef ndarray[np.uint8_t, ndim=2, cast=True] final = np.empty((nr, nc), dtype='bool')

    for i in range(nr):
        for j in range(nc):
            try:
                final[i, j] = arr[j] > arr2[i, j]
            except:
                final[i, j] = False
    return final

def ge_str(dict str_reverse_map, str other, ndarray[np.uint32_t, ndim=2] arr_codes):
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

def ge_str_one(ndarray[object] arr, ndarray[object] arr2):
    cdef int i, j
    cdef int nr = arr.shape[0]
    cdef ndarray[np.uint8_t, cast=True] final = np.empty(nr, dtype='bool')

    for i in range(nr):
        try:
            final[i] = arr[i] >= arr2[i]
        except:
            final[i] = False
    return final

def ge_str_two(ndarray[object, ndim=2] arr, ndarray[object, ndim=2] arr2):
    cdef int i, j
    cdef int nr = arr.shape[0]
    cdef int nc = arr.shape[1]
    cdef ndarray[np.uint8_t, ndim=2, cast=True] final = np.empty((nr, nc), dtype='bool')
    for i in range(nr):
        for j in range(nc):
            try:
                final[i, j] = arr[i, j] >= arr2[i, j]
            except:
                final[i, j] = False
    return final

def ge_str_two_1row_right(ndarray[object, ndim=2] arr, ndarray[object] arr2):
    cdef int i, j
    cdef int nr = arr.shape[0]
    cdef int nc = arr.shape[1]
    cdef ndarray[np.uint8_t, ndim=2, cast=True] final = np.empty((nr, nc), dtype='bool')

    for i in range(nr):
        for j in range(nc):
            try:
                final[i, j] = arr[i, j] >= arr2[j]
            except:
                final[i, j] = False
    return final

def ge_str_two_1row_left(ndarray[object] arr, ndarray[object, ndim=2] arr2):
    cdef int i, j
    cdef int nr = arr2.shape[0]
    cdef int nc = arr2.shape[1]
    cdef ndarray[np.uint8_t, ndim=2, cast=True] final = np.empty((nr, nc), dtype='bool')

    for i in range(nr):
        for j in range(nc):
            try:
                final[i, j] = arr[j] >= arr2[i, j]
            except:
                final[i, j] = False
    return final

def eq_str(dict str_reverse_map, str other, ndarray[np.uint32_t, ndim=2] arr_codes):
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

def eq_str_one(ndarray[object] arr, ndarray[object] arr2):
    cdef int i, j
    cdef int nr = arr.shape[0]
    cdef ndarray[np.uint8_t, cast=True] final = np.empty(nr, dtype='bool')

    for i in range(nr):
        try:
            final[i] = arr[i] == arr2[i]
        except:
            final[i] = False
    return final

def eq_str_two(ndarray[object, ndim=2] arr, ndarray[object, ndim=2] arr2):
    cdef int i, j
    cdef int nr = arr.shape[0]
    cdef int nc = arr.shape[1]
    cdef ndarray[np.uint8_t, ndim=2, cast=True] final = np.empty((nr, nc), dtype='bool')
    for i in range(nr):
        for j in range(nc):
            try:
                final[i, j] = arr[i, j] == arr2[i, j]
            except:
                final[i, j] = False
    return final

def eq_str_two_1row_right(ndarray[object, ndim=2] arr, ndarray[object] arr2):
    cdef int i, j
    cdef int nr = arr.shape[0]
    cdef int nc = arr.shape[1]
    cdef ndarray[np.uint8_t, ndim=2, cast=True] final = np.empty((nr, nc), dtype='bool')

    for i in range(nr):
        for j in range(nc):
            try:
                final[i, j] = arr[i, j] >= arr2[j]
            except:
                final[i, j] = False
    return final

def eq_str_two_1row_left(ndarray[object] arr, ndarray[object, ndim=2] arr2):
    cdef int i, j
    cdef int nr = arr2.shape[0]
    cdef int nc = arr2.shape[1]
    cdef ndarray[np.uint8_t, ndim=2, cast=True] final = np.empty((nr, nc), dtype='bool')

    for i in range(nr):
        for j in range(nc):
            try:
                final[i, j] = arr[j] == arr2[i, j]
            except:
                final[i, j] = False
    return final

def ne_str(dict str_reverse_map, str other, ndarray[np.uint32_t, ndim=2] arr_codes):
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

def ne_str_one(ndarray[object] arr, ndarray[object] arr2):
    cdef int i, j
    cdef int nr = arr.shape[0]
    cdef ndarray[np.uint8_t, cast=True] final = np.empty(nr, dtype='bool')

    for i in range(nr):
        try:
            final[i] = arr[i] != arr2[i]
        except:
            final[i] = False
    return final

def ne_str_two(ndarray[object, ndim=2] arr, ndarray[object, ndim=2] arr2):
    cdef int i, j
    cdef int nr = arr.shape[0]
    cdef int nc = arr.shape[1]
    cdef ndarray[np.uint8_t, ndim=2, cast=True] final = np.empty((nr, nc), dtype='bool')
    for i in range(nr):
        for j in range(nc):
            try:
                final[i, j] = arr[i, j] != arr2[i, j]
            except:
                final[i, j] = False
    return final

def ne_str_two_1row_right(ndarray[object, ndim=2] arr, ndarray[object] arr2):
    cdef int i, j
    cdef int nr = arr.shape[0]
    cdef int nc = arr.shape[1]
    cdef ndarray[np.uint8_t, ndim=2, cast=True] final = np.empty((nr, nc), dtype='bool')

    for i in range(nr):
        for j in range(nc):
            try:
                final[i, j] = arr[i, j] != arr2[j]
            except:
                final[i, j] = False
    return final

def ne_str_two_1row_left(ndarray[object] arr, ndarray[object, ndim=2] arr2):
    cdef int i, j
    cdef int nr = arr2.shape[0]
    cdef int nc = arr2.shape[1]
    cdef ndarray[np.uint8_t, ndim=2, cast=True] final = np.empty((nr, nc), dtype='bool')

    for i in range(nr):
        for j in range(nc):
            try:
                final[i, j] = arr[j] != arr2[i, j]
            except:
                final[i, j] = False
    return final
