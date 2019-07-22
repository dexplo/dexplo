#cython: boundscheck=False
#cython: wraparound=False
import numpy as np
cimport numpy as np
from numpy cimport ndarray

from copy import deepcopy

# arithmetic and comparison operations for strings

try:
    import bottleneck as bn
except ImportError:
    import numpy as bn

MIN_INT = np.iinfo('int64').min

def str__add__(dict str_reverse_map, ndarray[np.uint32_t, ndim=2] a, str other):
    cdef:
        Py_ssize_t i
        int n
        dict new_str_reverse_map = {}
        list new_list, cur_srm

    for loc, cur_srm in str_reverse_map.items():
        new_list = [False]
        new_str_reverse_map[loc] = new_list
        n = len(cur_srm)
        for i in range(1, n):
            new_list.append(cur_srm[i] + other)
    return new_str_reverse_map, a.copy('F'), 'S'

def str__radd__(dict str_reverse_map, ndarray[np.uint32_t, ndim=2] a, str other):
    cdef:
        Py_ssize_t i
        int n
        dict new_str_reverse_map = {}
        list new_list, cur_srm

    for loc, cur_srm in str_reverse_map.items():
        new_list = [False]
        new_str_reverse_map[loc] = new_list
        n = len(cur_srm)
        for i in range(1, n):
            new_list.append(other + cur_srm[i])
    return new_str_reverse_map, a.copy('F'), 'S'

def str__add__arr(ndarray[np.uint32_t] arr1, ndarray[np.uint32_t] arr2,
                  list srm1, list srm2):
    cdef:
        Py_ssize_t i, j
        int nr = len(arr1), r=len(srm1), c=len(srm2), left_code, right_code, final_code
        list new_str_reverse_map = [False], str_combos
        dict new_str_map = {False: 0}
        ndarray[np.uint32_t] arr = np.zeros(nr, 'uint32', 'F')

    if r * c < nr / 10:
        str_combos = []
        for i in range(1, r):
            for j in range(1, c):
                str_combos.append(srm1[i] + srm2[j])

        for i in range(nr):
            left_code = arr1[i] - 1
            right_code = arr2[i] - 1
            if left_code != -1 and right_code != -1:
                val = str_combos[left_code * (r - 1) + right_code]
                final_code = new_str_map.get(val, -1)
            if final_code == -1:
                new_str_reverse_map.append(val)
                arr[i] = len(new_str_map)
                new_str_map[val] = len(new_str_map)
            else:
                arr[i] = final_code
    else:
        for i in range(nr):
            left_code = arr1[i]
            right_code = arr2[i]
            if left_code != 0 and right_code != 0:
                val = srm1[left_code] + srm2[right_code]
                code = new_str_map.get(val, -1)
                if code == -1:
                    new_str_reverse_map.append(val)
                    arr[i] = len(new_str_map)
                    new_str_map[val] = len(new_str_map)
                else:
                    arr[i] = code

    return new_str_reverse_map, arr, 'S'

def str__radd__arr(ndarray[np.uint32_t] arr1, ndarray[np.uint32_t] arr2, list srm1, list srm2):
    return str__add__arr(arr2, arr1,  srm2, srm1)

def str__mul__(dict str_reverse_map, ndarray[np.uint32_t, ndim=2] a, int other):
    cdef:
        Py_ssize_t i
        int n
        dict new_str_reverse_map = {}
        list new_list, cur_srm
        bint has_empty_str
        ndarray[np.uint32_t, ndim=2] b

    if other == 0:
        for loc, cur_srm in str_reverse_map.items():
            new_list = [False]
            new_str_reverse_map[loc] = new_list
            if len(cur_srm) != 0:
                new_list.append('')
        b = a.clip(max=1)
    elif other == 1:
        new_str_reverse_map = deepcopy(str_reverse_map)
        b = a.copy('F')
    elif other < 0:
        raise ValueError('You cannot multiply a string by a negative number')
    else:
        for loc, cur_srm in str_reverse_map.items():
            new_list = [False]
            new_str_reverse_map[loc] = new_list
            n = len(cur_srm)
            for i in range(1, n):
                new_list.append(cur_srm[i] * other)
        b = a.copy('F')
    return new_str_reverse_map, b, 'S'

def str__rmul__(dict str_reverse_map, ndarray[np.uint32_t, ndim=2] a, int other):
    return str__mul__(str_reverse_map, a, other)

def str__mul__arr(ndarray[np.uint32_t] arr1, ndarray[np.int64_t] arr2, list srm1, list srm2):
    cdef:
        Py_ssize_t i
        int nr = arr1.shape[0], empty_loc, code
        ndarray[np.uint32_t] result = np.zeros(nr, 'uint32', 'F')
        list new_str_reverse_map = [False]
        dict new_str_map = {False: 0}
        str cur_val

    for i in range(nr):
        if arr1[i] != 0 and arr2[i] != MIN_INT:
            cur_val = srm1[arr1[i]] * arr2[i]
            final_code = new_str_map.get(cur_val, -1)
            if final_code == -1:
                new_str_reverse_map.append(cur_val)
                result[i] = len(new_str_map)
                new_str_map[cur_val] = len(new_str_map)
            else:
                result[i] = final_code
    return new_str_reverse_map, result, 'S'

def str__rmul__arr(ndarray[np.int64_t] arr1, ndarray[np.uint32_t] arr2, list srm1, list srm2):
    return str__mul__arr(arr2, arr1, srm2, srm1)

def str__lt__(dict str_reverse_map, ndarray[np.uint32_t, ndim=2] a, str other):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0], nc = a.shape[1], n, code
        ndarray[np.int8_t, ndim=2] final = np.empty((nr, nc), dtype='int8', order='F')
        ndarray[np.int8_t] b
        list cur_srm

    for i in range(nc):
        cur_srm = str_reverse_map[i]
        n = len(cur_srm)
        b = np.full(n, 0, 'int8')
        b[0] = -1
        for j in range(1, n):
            if cur_srm[j] < other:
                b[j] = 1
        for j in range(nr):
            final[j, i] = b[a[j, i]]

    return {}, final, 'b'

def str__lt__arr(ndarray[np.uint32_t] arr1, ndarray[np.uint32_t] arr2, list srm1, list srm2):
    cdef:
        Py_ssize_t i, j
        int nr = len(arr1), r=len(srm1), c=len(srm2), left_code, right_code, final_code
        ndarray[np.int8_t] result = np.full(nr, -1, 'int8', 'F')
        ndarray[np.int8_t, ndim=2] combos

    if r * c < nr / 10:
        combos = np.empty((r - 1, c - 1), 'int8', 'F')
        for i in range(1, r):
            for j in range(1, c):
                combos[i - 1, j - 1] = (srm1[i] < srm2[j]) * 1

        for i in range(nr):
            left_code = arr1[i] - 1
            right_code = arr2[i] - 1
            if left_code != -1 and right_code != -1:
                result[i] = combos[left_code, right_code]
    else:
        for i in range(nr):
            left_code = arr1[i]
            right_code = arr2[i]
            if left_code != 0 and right_code != 0:
                result[i] = (srm1[left_code] < srm2[right_code]) * 1

    return {}, result, 'b'

def str__le__(dict str_reverse_map, ndarray[np.uint32_t, ndim=2] a, str other):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0], nc = a.shape[1], n, code
        ndarray[np.int8_t, ndim=2] final = np.empty((nr, nc), dtype='int8', order='F')
        ndarray[np.int8_t] b
        list cur_srm

    for i in range(nc):
        cur_srm = str_reverse_map[i]
        n = len(cur_srm)
        b = np.full(n, 0, 'int8')
        b[0] = -1
        for j in range(1, n):
            if cur_srm[j] <= other:
                b[j] = 1
        for j in range(nr):
            final[j, i] = b[a[j, i]]

    return {}, final, 'b'

def str__le__arr(ndarray[np.uint32_t] arr1, ndarray[np.uint32_t] arr2, list srm1, list srm2):
    cdef:
        Py_ssize_t i, j
        int nr = len(arr1), r=len(srm1), c=len(srm2), left_code, right_code, final_code
        ndarray[np.int8_t] result = np.full(nr, -1, 'int8', 'F')
        ndarray[np.int8_t, ndim=2] combos

    if r * c < nr / 10:
        combos = np.empty((r - 1, c - 1), 'int8', 'F')
        for i in range(1, r):
            for j in range(1, c):
                combos[i - 1, j - 1] = (srm1[i] <= srm2[j]) * 1

        for i in range(nr):
            left_code = arr1[i] - 1
            right_code = arr2[i] - 1
            if left_code != -1 and right_code != -1:
                result[i] = combos[left_code, right_code]
    else:
        for i in range(nr):
            left_code = arr1[i]
            right_code = arr2[i]
            if left_code != 0 and right_code != 0:
                result[i] = (srm1[left_code] <= srm2[right_code]) * 1

    return {}, result, 'b'

def str__gt__(dict str_reverse_map, ndarray[np.uint32_t, ndim=2] a, str other):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0], nc = a.shape[1], n, code
        ndarray[np.int8_t, ndim=2] final = np.empty((nr, nc), dtype='int8', order='F')
        ndarray[np.int8_t] b
        list cur_srm

    for i in range(nc):
        cur_srm = str_reverse_map[i]
        n = len(cur_srm)
        b = np.full(n, 0, 'int8')
        b[0] = -1
        for j in range(1, n):
            if cur_srm[j] > other:
                b[j] = 1
        for j in range(nr):
            final[j, i] = b[a[j, i]]

    return {}, final, 'b'

def str__gt__arr(ndarray[np.uint32_t] arr1, ndarray[np.uint32_t] arr2, list srm1, list srm2):
    cdef:
        Py_ssize_t i, j
        int nr = len(arr1), r=len(srm1), c=len(srm2), left_code, right_code, final_code
        ndarray[np.int8_t] result = np.full(nr, -1, 'int8', 'F')
        ndarray[np.int8_t, ndim=2] combos

    if r * c < nr / 10:
        combos = np.empty((r - 1, c - 1), 'int8', 'F')
        for i in range(1, r):
            for j in range(1, c):
                combos[i - 1, j - 1] = (srm1[i] > srm2[j]) * 1

        for i in range(nr):
            left_code = arr1[i] - 1
            right_code = arr2[i] - 1
            if left_code != -1 and right_code != -1:
                result[i] = combos[left_code, right_code]
    else:
        for i in range(nr):
            left_code = arr1[i]
            right_code = arr2[i]
            if left_code != 0 and right_code != 0:
                result[i] = (srm1[left_code] > srm2[right_code]) * 1

    return {}, result, 'b'

def str__ge__(dict str_reverse_map, ndarray[np.uint32_t, ndim=2] a, str other):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0], nc = a.shape[1], n, code
        ndarray[np.int8_t, ndim=2] final = np.empty((nr, nc), dtype='int8', order='F')
        ndarray[np.int8_t] b
        list cur_srm

    for i in range(nc):
        cur_srm = str_reverse_map[i]
        n = len(cur_srm)
        b = np.full(n, 0, 'int8')
        b[0] = -1
        for j in range(1, n):
            if cur_srm[j] >= other:
                b[j] = 1
        for j in range(nr):
            final[j, i] = b[a[j, i]]

    return {}, final, 'b'

def str__ge__arr(ndarray[np.uint32_t] arr1, ndarray[np.uint32_t] arr2, list srm1, list srm2):
    cdef:
        Py_ssize_t i, j
        int nr = len(arr1), r=len(srm1), c=len(srm2), left_code, right_code, final_code
        ndarray[np.int8_t] result = np.full(nr, -1, 'int8', 'F')
        ndarray[np.int8_t, ndim=2] combos

    if r * c < nr / 10:
        combos = np.empty((r - 1, c - 1), 'int8', 'F')
        for i in range(1, r):
            for j in range(1, c):
                combos[i - 1, j - 1] = (srm1[i] >= srm2[j]) * 1

        for i in range(nr):
            left_code = arr1[i] - 1
            right_code = arr2[i] - 1
            if left_code != -1 and right_code != -1:
                result[i] = combos[left_code, right_code]
    else:
        for i in range(nr):
            left_code = arr1[i]
            right_code = arr2[i]
            if left_code != 0 and right_code != 0:
                result[i] = (srm1[left_code] >= srm2[right_code]) * 1

    return {}, result, 'b'

def str__eq__(dict str_reverse_map, ndarray[np.uint32_t, ndim=2] a, str other):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0], nc = a.shape[1], n
        ndarray[np.int8_t, ndim=2] final = np.empty((nr, nc), dtype='int8', order='F')
        ndarray[np.int8_t] b
        list cur_srm

    for i in range(nc):
        cur_srm = str_reverse_map[i]
        n = len(cur_srm)
        b = np.full(n, 0, 'int8')
        b[0] = -1
        for j in range(1, n):
            if cur_srm[j] == other:
                b[j] = 1
        for j in range(nr):
            final[j, i] = b[a[j, i]]

    return {}, final, 'b'

def str__eq__arr(ndarray[np.uint32_t] arr1, ndarray[np.uint32_t] arr2, list srm1, list srm2):
    cdef:
        Py_ssize_t i, j
        int nr = len(arr1), r=len(srm1), c=len(srm2), left_code, right_code, final_code
        ndarray[np.int8_t] result = np.full(nr, -1, 'int8', 'F')
        ndarray[np.int8_t, ndim=2] combos

    if r * c < nr / 10:
        combos = np.empty((r - 1, c - 1), 'int8', 'F')
        for i in range(1, r):
            for j in range(1, c):
                combos[i - 1, j - 1] = (srm1[i] == srm2[j]) * 1

        for i in range(nr):
            left_code = arr1[i] - 1
            right_code = arr2[i] - 1
            if left_code != -1 and right_code != -1:
                result[i] = combos[left_code, right_code]
    else:
        for i in range(nr):
            left_code = arr1[i]
            right_code = arr2[i]
            if left_code != 0 and right_code != 0:
                result[i] = (srm1[left_code] == srm2[right_code]) * 1

    return {}, result, 'b'

def str__ne__(dict str_reverse_map, ndarray[np.uint32_t, ndim=2] a, str other):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0], nc = a.shape[1], n
        ndarray[np.int8_t, ndim=2] final = np.empty((nr, nc), dtype='int8', order='F')
        ndarray[np.int8_t] b
        list cur_srm

    for i in range(nc):
        cur_srm = str_reverse_map[i]
        n = len(cur_srm)
        b = np.full(n, 0, 'int8')
        b[0] = -1
        for j in range(1, n):
            if cur_srm[j] != other:
                b[j] = 1
        for j in range(nr):
            final[j, i] = b[a[j, i]]

    return {}, final, 'b'

def str__ne__arr(ndarray[np.uint32_t] arr1, ndarray[np.uint32_t] arr2, list srm1, list srm2):
    cdef:
        Py_ssize_t i, j
        int nr = len(arr1), r=len(srm1), c=len(srm2), left_code, right_code, final_code
        ndarray[np.int8_t] result = np.full(nr, -1, 'int8', 'F')
        ndarray[np.int8_t, ndim=2] combos

    if r * c < nr / 10:
        combos = np.empty((r - 1, c - 1), 'int8', 'F')
        for i in range(1, r):
            for j in range(1, c):
                combos[i - 1, j - 1] = (srm1[i] != srm2[j]) * 1

        for i in range(nr):
            left_code = arr1[i] - 1
            right_code = arr2[i] - 1
            if left_code != -1 and right_code != -1:
                result[i] = combos[left_code, right_code]
    else:
        for i in range(nr):
            left_code = arr1[i]
            right_code = arr2[i]
            if left_code != 0 and right_code != 0:
                result[i] = (srm1[left_code] != srm2[right_code]) * 1

    return {}, result, 'b'
