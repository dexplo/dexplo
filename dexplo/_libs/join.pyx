import numpy as np
cimport numpy as np
from numpy cimport ndarray
from numpy import nan
from cython cimport list
from collections import defaultdict
cimport cython


# def join_str_1d(ndarray[object] left, ndarray[object] right):
#     cdef:
#         Py_ssize_t i, n_right=len(right), n_left=len(left), n_new=0, idx, ct=0, cur_len
#         ndarray[np.int64_t] right_idx
#         ndarray[np.int64_t] left_rep = np.empty(n_left, dtype=np.int64)
#
#     right_idx_list = defaultdict(list)
#
#     for i in range(n_right):
#         right_idx_list[right[i]].append(i)
#
#     for i in range(n_left):
#         cur_len = len(right_idx_list[left[i]])
#         left_rep[i] = cur_len
#         n_new += cur_len
#
#     right_idx = np.empty(n_new, dtype='int64')
#
#     for i in range(n_left):
#         for idx in right_idx_list[left[i]]:
#             right_idx[ct] = idx
#             ct += 1
#
#     return left_rep, right_idx, n_new
#
# @cython.wraparound(False)
# @cython.boundscheck(False)
def join_str_1d(ndarray[object] left, ndarray[object] right):
    cdef:
        Py_ssize_t i, j, n_right=len(right), n_left=len(left), n_new=0, ct=0, cur_ct, count=0, max_ct, g
        ndarray[np.int64_t] left_rep = np.empty(n_left, dtype=np.int64)
        ndarray[np.int64_t] group_right = np.empty(n_right, dtype=np.int64)
        ndarray[np.int64_t] group_left = np.empty(n_left, dtype=np.int64)
        ndarray[np.int64_t] group_cts = np.empty(n_right, dtype=np.int64)
        ndarray[np.int64_t] right_idx = np.empty(n_right, dtype=np.int64)
        ndarray[np.int64_t] right_cur_ct
        ndarray[np.int64_t, ndim=2] right_idx_matrix
        dict d = {}

    for i in range(n_right):
        group_right[i] = d.get(right[i], -1)
        if group_right[i] == -1:
            group_right[i] = count
            d[right[i]] = count
            group_cts[count] = 1
            count += 1
        else:
            group_cts[group_right[i]] += 1

    right_cur_ct = np.zeros(count, dtype='int64')
    max_ct = group_cts[:count].max()
    right_idx_matrix = np.empty((count, max_ct), dtype='int64')

    for i in range(n_right):
        j = right_cur_ct[group_right[i]]
        right_idx_matrix[group_right[i], j] = i
        right_cur_ct[group_right[i]] += 1

    for i in range(n_left):
        group_left[i] = d.get(left[i], -1)
        if group_left[i] != -1:
            left_rep[i] = group_cts[group_left[i]]
            n_new += left_rep[i]
        else:
            left_rep[i] = 0

    right_idx = np.empty(n_new, dtype='int64')

    for i in range(n_left):
        g = group_left[i]
        if g != -1:
            cur_ct = group_cts[g]
            for j in range(cur_ct):
                right_idx[ct] = right_idx_matrix[g, j]
                ct += 1

    return left_rep, right_idx, n_new


def join_int_1d(ndarray[np.int64_t] left, ndarray[np.int64_t] right):
    cdef:
        Py_ssize_t i, n_right=len(right), n_left=len(left), n_new=0, idx, ct=0, cur_len
        ndarray[np.int64_t] right_idx
        ndarray[np.int64_t] left_rep = np.empty(n_left, dtype=np.int64)

    right_idx_list = defaultdict(list)

    for i in range(n_right):
        right_idx_list[right[i]].append(i)

    for i in range(n_left):
        cur_len = len(right_idx_list[left[i]])
        left_rep[i] = cur_len
        n_new += cur_len

    right_idx = np.empty(n_new, dtype='int64')

    for i in range(n_left):
        for idx in right_idx_list[left[i]]:
            right_idx[ct] = idx
            ct += 1

    return left_rep, right_idx, n_new

def join_float_1d(ndarray[np.float64_t] left, ndarray[np.float64_t] right):
    cdef:
        Py_ssize_t i, n_right=len(right), n_left=len(left), n_new=0, idx, ct=0, cur_len
        ndarray[np.int64_t] right_idx
        ndarray[np.int64_t] left_rep = np.empty(n_left, dtype=np.int64)

    right_idx_list = defaultdict(list)

    for i in range(n_right):
        right_idx_list[right[i]].append(i)

    for i in range(n_left):
        cur_len = len(right_idx_list[left[i]])
        left_rep[i] = cur_len
        n_new += cur_len

    right_idx = np.empty(n_new, dtype='int64')

    for i in range(n_left):
        for idx in right_idx_list[left[i]]:
            right_idx[ct] = idx
            ct += 1

    return left_rep, right_idx, n_new


def join_bool_1d(ndarray[bint] left, ndarray[bint] right):
    cdef:
        Py_ssize_t i, n_right=len(right), n_left=len(left), n_new=0, idx, ct=0, cur_len
        ndarray[np.int64_t] right_idx
        ndarray[np.int64_t] left_rep = np.empty(n_left, dtype=np.int64)

    right_idx_list = defaultdict(list)

    for i in range(n_right):
        right_idx_list[right[i]].append(i)

    for i in range(n_left):
        cur_len = len(right_idx_list[left[i]])
        left_rep[i] = cur_len
        n_new += cur_len

    right_idx = np.empty(n_new, dtype='int64')

    for i in range(n_left):
        for idx in right_idx_list[left[i]]:
            right_idx[ct] = idx
            ct += 1

    return left_rep, right_idx, n_new

# def join_str_2d(ndarray[object, ndim=2] left, ndarray[object, ndim=2] right,
#                 list left_locs, list right_locs):
#     cdef:
#         Py_ssize_t i, j, nr_right=len(right), nr_left=len(left), nc_right = len(right_locs), nc_left = len(left_locs)
#         Py_ssize_t n_new=0, idx, ct=0, cur_len
#         ndarray[np.int64_t] right_idx
#         ndarray[np.int64_t] left_rep = np.empty(nr_left, dtype=np.int64)
#         list vals = list(range(nc_right))
#         tuple t
#         list temp_list
#
#     right_idx_list = defaultdict(list)
#
#     for i in range(nr_right):
#         for j in range(nc_right):
#             vals[j] = right[i, right_locs[j]]
#         t = tuple(vals)
#         right_idx_list[t].append(i)
#
#     for i in range(nr_left):
#         for j in range(nc_left):
#             vals[j] = left[i, left_locs[j]]
#         t = tuple(vals)
#         cur_len = len(right_idx_list[t])
#         left_rep[i] = cur_len
#         n_new += cur_len
#
#     right_idx = np.empty(n_new, dtype='int64')
#
#     for i in range(nr_left):
#         for j in range(nc_left):
#             vals[j] = left[i, left_locs[j]]
#         t = tuple(vals)
#         temp_list = right_idx_list[t]
#         cur_len = len(temp_list)
#         for j in range(cur_len):
#             right_idx[ct] = temp_list[j]
#             ct += 1
#
#     return left_rep, right_idx, n_new

def join_str_2d(ndarray[object, ndim=2] left, ndarray[object, ndim=2] right,
                list left_locs, list right_locs):
    cdef:
        Py_ssize_t i, j, nr_right=len(right), nr_left=len(left), nc_right = len(right_locs), nc_left = len(left_locs)
        Py_ssize_t n_new=0, idx, ct=0, cur_len
        ndarray[np.int64_t] right_idx
        ndarray[np.int64_t] left_rep = np.empty(nr_left, dtype=np.int64)
        list vals = list(range(nc_right))
        tuple t
        list temp_list
        ndarray[object] left_tuples = np.empty(nr_left, dtype='O')
        ndarray[object] right_tuples = np.empty(nr_right, dtype='O')
        ndarray[object] right_arrays = np.empty(nr_right, dtype='O')

    right_idx_list = defaultdict(list)

    for i in range(nr_right):
        for j in range(nc_right):
            vals[j] = right[i, right_locs[j]]
        t = tuple(vals)
        right_tuples[i] = t

    for i in range(nr_right):
        right_arrays[i] = np.empty(len(right_tuples[i]), dtype='int64')

    for i in range(nr_left):
        for j in range(nc_left):
            vals[j] = left[i, left_locs[j]]
        t = tuple(vals)
        cur_len = len(right_idx_list[t])
        left_rep[i] = cur_len
        n_new += cur_len
        left_tuples[i] = t

    right_idx = np.empty(n_new, dtype='int64')

    for i in range(nr_left):
        t = left_tuples[i]
        temp_list = right_idx_list[t]
        cur_len = len(temp_list)
        for j in range(cur_len):
            right_idx[ct] = temp_list[j]
            ct += 1

    return left_rep, right_idx, n_new