#cython: boundscheck=False
#cython: wraparound=False
import numpy as np
cimport numpy as np
from numpy cimport ndarray
import cython
from numpy import nan
from .math import var_int as var_int_math
from collections import defaultdict

try:
    import bottleneck as bn
except ImportError:
    import numpy as bn

cdef np.float64_t MAX_FLOAT = np.finfo(np.float64).max
cdef np.float64_t MIN_FLOAT = np.finfo(np.float64).min

cdef np.int64_t MAX_INT = np.iinfo(np.int64).max
cdef np.int64_t MIN_INT = np.iinfo(np.int64).min

MAX_CHAR = chr(1_000_000)
MIN_CHAR = chr(0)


cdef extern from "numpy/npy_math.h":
    bint npy_isnan(double x)


def sum_int(ndarray[np.int64_t, ndim=2] a, ndarray[np.int64_t] locs,
            int left, int right, int min_window):
    cdef:
        Py_ssize_t i=0, j, k, j_act, start, middle, end, ct = 0
        Py_ssize_t nr = a.shape[0], nc = a.shape[1], nc_actual = len(locs)
        ndarray[np.int64_t, ndim=2] result = np.zeros((nr, nc_actual), dtype='int64')
        ndarray[np.int64_t] temp
        np.int64_t total = 0
        bint not_started = True


    if left < 0:
        start = -left + 1
        end = nr
    else:
        end = nr - left
        start = 1

    if right > 0:
        middle = nr - right
    else:
        middle = nr

    for j in range(nc_actual):

        j_act = locs[j]
        temp = a[:, j_act]

        # must enter first for loop before subtraction can begin
        for i in range(start):
            if not_started:
                if i + right > 0:
                    not_started = False
                    for k in range(max(0, i + left), i + right):
                        total += temp[k]
                else:
                    continue
            else:
                total += temp[i + right - 1]
            result[i, j] = total

        for i in range(start, middle):
            total = total + temp[i + right - 1] - temp[i + left - 1]
            result[i, j] = total

        if middle != nr:
            total += temp[i + right]

        for i in range(middle, end):
            total -= temp[i + left - 1]
            result[i, j] = total

    return result

def sum_float(ndarray[np.float64_t, ndim=2] a, ndarray[np.int64_t] locs,
            int left, int right, int min_window):
    cdef:
        Py_ssize_t i=0, j, k, j_act, start, middle, end, ct = 0
        Py_ssize_t nr = a.shape[0], nc = a.shape[1], nc_actual = len(locs)
        ndarray[np.float64_t, ndim=2] result = np.zeros((nr, nc_actual), dtype='float64')
        ndarray[np.float64_t] temp
        np.float64_t total = 0
        bint not_started = True


    if left < 0:
        start = -left + 1
        end = nr
    else:
        end = nr - left
        start = 1

    if right > 0:
        middle = nr - right
    else:
        middle = nr

    for j in range(nc_actual):

        j_act = locs[j]
        temp = a[:, j_act]

        # must enter first for loop before subtraction can begin
        for i in range(start):
            if not_started:
                if i + right > 0:
                    not_started = False
                    for k in range(max(0, i + left), i + right):
                        total += temp[k]
                else:
                    continue
            else:
                total += temp[i + right - 1]
            result[i, j] = total

        for i in range(start, middle):
            total = total + temp[i + right - 1] - temp[i + left - 1]
            result[i, j] = total

        if middle != nr:
            total += temp[i + right]

        for i in range(middle, end):
            total -= temp[i + left - 1]
            result[i, j] = total

    return result

def sum_bool(ndarray[np.uint8_t, ndim=2, cast=True] a, ndarray[np.int64_t] locs,
            int left, int right, int min_window):
    cdef:
        Py_ssize_t i=0, j, k, j_act, start, middle, end, ct = 0
        Py_ssize_t nr = a.shape[0], nc = a.shape[1], nc_actual = len(locs)
        ndarray[np.int64_t, ndim=2] result = np.zeros((nr, nc_actual), dtype='int64')
        ndarray[np.uint8_t, cast=True] temp
        np.int64_t total = 0
        bint not_started = True


    if left < 0:
        start = -left + 1
        end = nr
    else:
        end = nr - left
        start = 1

    if right > 0:
        middle = nr - right
    else:
        middle = nr

    for j in range(nc_actual):

        j_act = locs[j]
        temp = a[:, j_act]

        # must enter first for loop before subtraction can begin
        for i in range(start):
            if not_started:
                if i + right > 0:
                    not_started = False
                    for k in range(max(0, i + left), i + right):
                        total += temp[k]
                else:
                    continue
            else:
                total += temp[i + right - 1]
            result[i, j] = total

        for i in range(start, middle):
            total = total + temp[i + right - 1] - temp[i + left - 1]
            result[i, j] = total

        if middle != nr:
            total += temp[i + right]

        for i in range(middle, end):
            total -= temp[i + left - 1]
            result[i, j] = total

    return result


def min_int(ndarray[np.int64_t, ndim=2] a, ndarray[np.int64_t] locs,
            int left, int right, int min_window):
    cdef:
        Py_ssize_t i=0, j, k, j_act, start, middle, end, ct = 0, idx
        Py_ssize_t nr = a.shape[0], nc = a.shape[1], nc_actual = len(locs)
        ndarray[np.int64_t, ndim=2] result = np.zeros((nr, nc_actual), dtype='int64')
        ndarray[np.int64_t] temp
        np.int64_t val
        bint not_started = True, first = True


    if left < 0:
        start = -left + 1
        end = nr
    else:
        end = nr - left
        start = 1

    if right > 0:
        middle = nr - right
    else:
        middle = nr

    for j in range(nc_actual):

        j_act = locs[j]
        temp = a[:, j_act]

        # must enter first for loop before subtraction can begin
        for i in range(start):
            if not_started:
                if i + right > 0:
                    not_started = False
                    for k in range(max(0, i + left), i + right):
                        if first:
                            first = False
                            val = temp[k]
                            idx = k
                        elif temp[k] < val:
                            val = temp[k]
                            idx = k
                else:
                    continue
            else:
                if temp[i + right - 1] < val:
                    val = temp[i + right - 1]
                    idx = i + right - 1
            result[i, j] = val

        for i in range(start, middle):
            if temp[i + right - 1] < val:
                val = temp[i + right - 1]
                idx = i + right - 1
            if idx == i + left - 1:
                val = temp[i + left]
                idx = i + left
                for k in range(i + left, i + right - 1):
                    if temp[k] < val:
                        val = temp[k]
                        idx = k
            result[i, j] = val

        for i in range(middle, end):
            if idx == i + left - 1:
                val = temp[i + left]
                idx = i + left
                for k in range(i + left, nr):
                    if temp[k] < val:
                        val = temp[k]
                        idx = k
            result[i, j] = val

    return result

def min_float(ndarray[np.float64_t, ndim=2] a, ndarray[np.int64_t] locs,
            int left, int right, int min_window):
    cdef:
        Py_ssize_t i=0, j, k, j_act, start, middle, end, ct = 0, idx
        Py_ssize_t nr = a.shape[0], nc = a.shape[1], nc_actual = len(locs)
        ndarray[np.float64_t, ndim=2] result = np.full((nr, nc_actual), nan, dtype='float64')
        ndarray[np.float64_t] temp
        np.float64_t val
        bint not_started = True, first = True

    if left < 0:
        start = -left + 1
        end = nr
    else:
        end = nr - left
        start = 1

    if right > 0:
        middle = nr - right
    else:
        middle = nr

    for j in range(nc_actual):

        j_act = locs[j]
        temp = a[:, j_act]

        # must enter first for loop before subtraction can begin
        for i in range(start):
            if not_started:
                if i + right > 0:
                    not_started = False
                    for k in range(max(0, i + left), i + right):
                        if first:
                            first = False
                            val = temp[k]
                            idx = k
                        elif temp[k] < val:
                            val = temp[k]
                            idx = k
                else:
                    continue
            else:
                if temp[i + right - 1] < val:
                    val = temp[i + right - 1]
                    idx = i + right - 1
            result[i, j] = val

        for i in range(start, middle):
            if temp[i + right - 1] < val:
                val = temp[i + right - 1]
                idx = i + right - 1
            if idx == i + left - 1:
                val = temp[i + left]
                idx = i + left
                for k in range(i + left, i + right - 1):
                    if temp[k] < val:
                        val = temp[k]
                        idx = k
            result[i, j] = val

        for i in range(middle, end):
            if idx == i + left - 1:
                val = temp[i + left]
                idx = i + left
                for k in range(i + left, nr):
                    if temp[k] < val:
                        val = temp[k]
                        idx = k
            result[i, j] = val

    return result

def min_bool(ndarray[np.uint8_t, ndim=2, cast=True] a, ndarray[np.int64_t] locs,
            int left, int right, int min_window):
    cdef:
        Py_ssize_t i=0, j, k, j_act, start, middle, end, ct = 0, idx
        Py_ssize_t nr = a.shape[0], nc = a.shape[1], nc_actual = len(locs)
        ndarray[np.uint8_t, ndim=2] result = np.zeros((nr, nc_actual), dtype='bool')
        ndarray[np.uint8_t] temp
        bint val
        bint not_started = True, first = True


    if left < 0:
        start = -left + 1
        end = nr
    else:
        end = nr - left
        start = 1

    if right > 0:
        middle = nr - right
    else:
        middle = nr

    for j in range(nc_actual):

        j_act = locs[j]
        temp = a[:, j_act]

        # must enter first for loop before subtraction can begin
        for i in range(start):
            if not_started:
                if i + right > 0:
                    not_started = False
                    for k in range(max(0, i + left), i + right):
                        if first:
                            first = False
                            val = temp[k]
                            idx = k
                        elif temp[k] < val:
                            val = temp[k]
                            idx = k
                else:
                    continue
            else:
                if temp[i + right - 1] < val:
                    val = temp[i + right - 1]
                    idx = i + right - 1
            result[i, j] = val

        for i in range(start, middle):
            if temp[i + right - 1] < val:
                val = temp[i + right - 1]
                idx = i + right - 1
            if idx == i + left - 1:
                val = temp[i + left]
                idx = i + left
                for k in range(i + left, i + right - 1):
                    if temp[k] < val:
                        val = temp[k]
                        idx = k
            result[i, j] = val

        for i in range(middle, end):
            if idx == i + left - 1:
                val = temp[i + left]
                idx = i + left
                for k in range(i + left, nr):
                    if temp[k] < val:
                        val = temp[k]
                        idx = k
            result[i, j] = val

    return result

def max_int(ndarray[np.int64_t, ndim=2] a, ndarray[np.int64_t] locs,
            int left, int right, int min_window):
    cdef:
        Py_ssize_t i=0, j, k, j_act, start, middle, end, ct = 0, idx
        Py_ssize_t nr = a.shape[0], nc = a.shape[1], nc_actual = len(locs)
        ndarray[np.int64_t, ndim=2] result = np.zeros((nr, nc_actual), dtype='int64')
        ndarray[np.int64_t] temp
        np.int64_t val
        bint not_started = True, first = True

    if left < 0:
        start = -left + 1
        end = nr
    else:
        end = nr - left
        start = 1

    if right > 0:
        middle = nr - right
    else:
        middle = nr

    for j in range(nc_actual):

        j_act = locs[j]
        temp = a[:, j_act]

        # must enter first for loop before subtraction can begin
        for i in range(start):
            if not_started:
                if i + right > 0:
                    not_started = False
                    for k in range(max(0, i + left), i + right):
                        if first:
                            first = False
                            val = temp[k]
                            idx = k
                        elif temp[k] > val:
                            val = temp[k]
                            idx = k
                else:
                    continue
            else:
                if temp[i + right - 1] > val:
                    val = temp[i + right - 1]
                    idx = i + right - 1
            result[i, j] = val

        for i in range(start, middle):
            if temp[i + right - 1] > val:
                val = temp[i + right - 1]
                idx = i + right - 1
            if idx == i + left - 1:
                val = temp[i + left]
                idx = i + left
                for k in range(i + left, i + right - 1):
                    if temp[k] < val:
                        val = temp[k]
                        idx = k
            result[i, j] = val

        for i in range(middle, end):
            if idx == i + left - 1:
                val = temp[i + left]
                idx = i + left
                for k in range(i + left, nr):
                    if temp[k] > val:
                        val = temp[k]
                        idx = k
            result[i, j] = val

    return result

def max_float(ndarray[np.float64_t, ndim=2] a, ndarray[np.int64_t] locs,
            int left, int right, int min_window):
    cdef:
        Py_ssize_t i=0, j, k, j_act, start, middle, end, ct = 0, idx
        Py_ssize_t nr = a.shape[0], nc = a.shape[1], nc_actual = len(locs)
        ndarray[np.float64_t, ndim=2] result = np.full((nr, nc_actual), nan, dtype='float64')
        ndarray[np.float64_t] temp
        np.float64_t val
        bint not_started = True, first = True

    if left < 0:
        start = -left + 1
        end = nr
    else:
        end = nr - left
        start = 1

    if right > 0:
        middle = nr - right
    else:
        middle = nr

    for j in range(nc_actual):

        j_act = locs[j]
        temp = a[:, j_act]

        # must enter first for loop before subtraction can begin
        for i in range(start):
            if not_started:
                if i + right > 0:
                    not_started = False
                    for k in range(max(0, i + left), i + right):
                        if first:
                            first = False
                            val = temp[k]
                            idx = k
                        elif temp[k] > val:
                            val = temp[k]
                            idx = k
                else:
                    continue
            else:
                if temp[i + right - 1] > val:
                    val = temp[i + right - 1]
                    idx = i + right - 1
            result[i, j] = val

        for i in range(start, middle):
            if temp[i + right - 1] > val:
                val = temp[i + right - 1]
                idx = i + right - 1
            if idx == i + left - 1:
                val = temp[i + left]
                idx = i + left
                for k in range(i + left, i + right - 1):
                    if temp[k] > val:
                        val = temp[k]
                        idx = k
            result[i, j] = val

        for i in range(middle, end):
            if idx == i + left - 1:
                val = temp[i + left]
                idx = i + left
                for k in range(i + left, nr):
                    if temp[k] > val:
                        val = temp[k]
                        idx = k
            result[i, j] = val

    return result

def max_bool(ndarray[np.uint8_t, ndim=2, cast=True] a, ndarray[np.int64_t] locs,
            int left, int right, int min_window):
    cdef:
        Py_ssize_t i=0, j, k, j_act, start, middle, end, ct = 0, idx
        Py_ssize_t nr = a.shape[0], nc = a.shape[1], nc_actual = len(locs)
        ndarray[np.uint8_t, ndim=2] result = np.zeros((nr, nc_actual), dtype='bool')
        ndarray[np.uint8_t] temp
        bint val
        bint not_started = True, first = True

    if left < 0:
        start = -left + 1
        end = nr
    else:
        end = nr - left
        start = 1

    if right > 0:
        middle = nr - right
    else:
        middle = nr

    for j in range(nc_actual):

        j_act = locs[j]
        temp = a[:, j_act]

        # must enter first for loop before subtraction can begin
        for i in range(start):
            if not_started:
                if i + right > 0:
                    not_started = False
                    for k in range(max(0, i + left), i + right):
                        if first:
                            first = False
                            val = temp[k]
                            idx = k
                        elif temp[k] > val:
                            val = temp[k]
                            idx = k
                else:
                    continue
            else:
                if temp[i + right - 1] > val:
                    val = temp[i + right - 1]
                    idx = i + right - 1
            result[i, j] = val

        for i in range(start, middle):
            if temp[i + right - 1] > val:
                val = temp[i + right - 1]
                idx = i + right - 1
            if idx == i + left - 1:
                val = temp[i + left]
                idx = i + left
                for k in range(i + left, i + right - 1):
                    if temp[k] > val:
                        val = temp[k]
                        idx = k
            result[i, j] = val

        for i in range(middle, end):
            if idx == i + left - 1:
                val = temp[i + left]
                idx = i + left
                for k in range(i + left, nr):
                    if temp[k] > val:
                        val = temp[k]
                        idx = k
            result[i, j] = val

    return result

def mean_int(ndarray[np.int64_t, ndim=2] a, ndarray[np.int64_t] locs,
            int left, int right, int min_window):
    cdef:
        Py_ssize_t i=0, j, k, j_act, start, middle, end, ct = 0, n, n1 = right - left
        Py_ssize_t nr = a.shape[0], nc = a.shape[1], nc_actual = len(locs)
        ndarray[np.float64_t, ndim=2] result = np.zeros((nr, nc_actual), dtype='float64')
        ndarray[np.int64_t] temp
        np.int64_t total = 0
        bint not_started = True

    if left < 0:
        start = -left + 1
        end = nr
    else:
        end = nr - left
        start = 1

    if right > 0:
        middle = nr - right
    else:
        middle = nr

    for j in range(nc_actual):

        j_act = locs[j]
        temp = a[:, j_act]

        # must enter first for loop before subtraction can begin
        for i in range(start):
            if not_started:
                if i + right > 0:
                    not_started = False
                    n = 0
                    for k in range(max(0, i + left), i + right):
                        total += temp[k]
                        n += 1
                else:
                    continue
            else:
                n += 1
                total += temp[i + right - 1]
            result[i, j] = total / n

        for i in range(start, middle):
            total = total + temp[i + right - 1] - temp[i + left - 1]
            result[i, j] = total / n1

        if middle != nr:
            total += temp[i + right]

        for i in range(middle, end):
            total -= temp[i + left - 1]
            n1 -= 1
            result[i, j] = total / n1

    return result

def mean_float(ndarray[np.float64_t, ndim=2] a, ndarray[np.int64_t] locs,
            int left, int right, int min_window):
    cdef:
        Py_ssize_t i=0, j, k, j_act, start, middle, end, ct = 0, n, n1 = right - left
        Py_ssize_t nr = a.shape[0], nc = a.shape[1], nc_actual = len(locs)
        ndarray[np.float64_t, ndim=2] result = np.zeros((nr, nc_actual), dtype='float64')
        ndarray[np.float64_t] temp
        np.float64_t total = 0
        bint not_started = True

    if left < 0:
        start = -left + 1
        end = nr
    else:
        end = nr - left
        start = 1

    if right > 0:
        middle = nr - right
    else:
        middle = nr

    for j in range(nc_actual):

        j_act = locs[j]
        temp = a[:, j_act]

        # must enter first for loop before subtraction can begin
        for i in range(start):
            if not_started:
                if i + right > 0:
                    not_started = False
                    n = 0
                    for k in range(max(0, i + left), i + right):
                        if not npy_isnan(temp[k]):
                            total += temp[k]
                            n += 1
                else:
                    continue
            else:
                if not npy_isnan(temp[k]):
                    total += temp[i + right - 1]
                    n += 1

            if n != 0:
                result[i, j] = total / n

        for i in range(start, middle):
            if not npy_isnan(temp[i + right - 1]):
                total += temp[i + right - 1]
                n += 1
            if not npy_isnan(temp[i + left - 1]):
                total -= temp[i + left - 1]
                n -= 1
            result[i, j] = total / n

        if middle != nr:
            if not npy_isnan(temp[i + right]):
                total += temp[i + right]
                n += 1

        for i in range(middle, end):
            if not npy_isnan(temp[i + left - 1]):
                total -= temp[i + left - 1]
                n -= 1
            result[i, j] = total / n

    return result

def mean_bool(ndarray[np.uint8_t, ndim=2, cast=True] a, ndarray[np.int64_t] locs,
            int left, int right, int min_window):
    cdef:
        Py_ssize_t i=0, j, k, j_act, start, middle, end, ct = 0, n, n1 = right - left
        Py_ssize_t nr = a.shape[0], nc = a.shape[1], nc_actual = len(locs)
        ndarray[np.float64_t, ndim=2] result = np.zeros((nr, nc_actual), dtype='float64')
        ndarray[np.uint8_t, cast=True] temp
        np.int64_t total = 0
        bint not_started = True

    if left < 0:
        start = -left + 1
        end = nr
    else:
        end = nr - left
        start = 1

    if right > 0:
        middle = nr - right
    else:
        middle = nr

    for j in range(nc_actual):

        j_act = locs[j]
        temp = a[:, j_act]

        # must enter first for loop before subtraction can begin
        for i in range(start):
            if not_started:
                if i + right > 0:
                    not_started = False
                    n = 0
                    for k in range(max(0, i + left), i + right):
                        total += temp[k]
                        n += 1
                else:
                    continue
            else:
                n += 1
                total += temp[i + right - 1]
            result[i, j] = total / n

        for i in range(start, middle):
            total = total + temp[i + right - 1] - temp[i + left - 1]
            result[i, j] = total / n1

        if middle != nr:
            total += temp[i + right]

        for i in range(middle, end):
            total -= temp[i + left - 1]
            n1 -= 1
            result[i, j] = total / n1

    return result

def count_int(ndarray[np.int64_t, ndim=2] a, ndarray[np.int64_t] locs,
            int left, int right, int min_window):
    cdef:
        Py_ssize_t i=0, j, k, j_act, start, middle, end, ct = 0, n, n1 = right - left
        Py_ssize_t nr = a.shape[0], nc = a.shape[1], nc_actual = len(locs)
        ndarray[np.int64_t, ndim=2] result = np.zeros((nr, nc_actual), dtype='int64')
        ndarray[np.int64_t] temp
        np.int64_t total = 0
        bint not_started = True

    if left < 0:
        start = -left + 1
        end = nr
    else:
        end = nr - left
        start = 1

    if right > 0:
        middle = nr - right
    else:
        middle = nr

    for j in range(nc_actual):

        j_act = locs[j]
        temp = a[:, j_act]

        # must enter first for loop before subtraction can begin
        for i in range(start):
            if not_started:
                if i + right > 0:
                    not_started = False
                    n = 0
                    for k in range(max(0, i + left), i + right):
                        total += temp[k]
                        n += 1
                else:
                    continue
            else:
                n += 1
                total += temp[i + right - 1]
            result[i, j] = n

        for i in range(start, middle):
            result[i, j] = n1

        if middle != nr:
            total += temp[i + right]

        for i in range(middle, end):
            n1 -= 1
            result[i, j] = n1

    return result

def count_float(ndarray[np.float64_t, ndim=2] a, ndarray[np.int64_t] locs,
            int left, int right, int min_window):

    cdef:
        Py_ssize_t i=0, j, k, j_act, start, middle, end, ct = 0, n=0, n1 = right - left
        Py_ssize_t nr = a.shape[0], nc = a.shape[1], nc_actual = len(locs)
        ndarray[np.int64_t, ndim=2] result = np.zeros((nr, nc_actual), dtype='int64')
        ndarray[np.float64_t] temp
        bint not_started = True

    if left < 0:
        start = -left + 1
        end = nr
    else:
        end = nr - left
        start = 1

    if right > 0:
        middle = nr - right
    else:
        middle = nr

    for j in range(nc_actual):

        j_act = locs[j]
        temp = a[:, j_act]

        # must enter first for loop before subtraction can begin
        for i in range(start):
            if not_started:
                if i + right > 0:
                    not_started = False
                    n = 0
                    for k in range(max(0, i + left), i + right):
                        if not npy_isnan(temp[k]):
                            n += 1
                else:
                    continue
            else:
                if not npy_isnan(temp[i + right - 1]):
                    n += 1
            if n != 0:
                result[i, j] = n

        for i in range(start, middle):
            if not npy_isnan(temp[i + right - 1]):
                n += 1
            if not npy_isnan(temp[i + left - 1]):
                n -= 1
            result[i, j] = n

        if middle != nr:
            if not npy_isnan(temp[i + right]):
                n += 1

        for i in range(middle, end):
            if not npy_isnan(temp[i + left - 1]):
                n -= 1
            result[i, j] = n

    return result

def count_bool(ndarray[np.uint8_t, ndim=2, cast=True] a, ndarray[np.int64_t] locs,
            int left, int right, int min_window):
    cdef:
        Py_ssize_t i=0, j, k, j_act, start, middle, end, ct = 0, n, n1 = right - left
        Py_ssize_t nr = a.shape[0], nc = a.shape[1], nc_actual = len(locs)
        ndarray[np.float64_t, ndim=2] result = np.zeros((nr, nc_actual), dtype='float64')
        ndarray[np.uint8_t, cast=True] temp
        bint not_started = True

    if left < 0:
        start = -left + 1
        end = nr
    else:
        end = nr - left
        start = 1

    if right > 0:
        middle = nr - right
    else:
        middle = nr

    for j in range(nc_actual):

        j_act = locs[j]
        temp = a[:, j_act]

        # must enter first for loop before subtraction can begin
        for i in range(start):
            if not_started:
                if i + right > 0:
                    not_started = False
                    n = 0
                    for k in range(max(0, i + left), i + right):
                        n += 1
                else:
                    continue
            else:
                n += 1
            result[i, j] = n

        for i in range(start, middle):
            result[i, j] = n1

        for i in range(middle, end):
            n1 -= 1
            result[i, j] = n1

    return result

def count_str(ndarray[object, ndim=2] a, ndarray[np.int64_t] locs,
            int left, int right, int min_window):
    cdef:
        Py_ssize_t i=0, j, k, j_act, start, middle, end, ct = 0, n, n1 = right - left
        Py_ssize_t nr = a.shape[0], nc = a.shape[1], nc_actual = len(locs)
        ndarray[np.int64_t, ndim=2] result = np.zeros((nr, nc_actual), dtype='int64')
        ndarray[object] temp
        bint not_started = True

    if left < 0:
        start = -left + 1
        end = nr
    else:
        end = nr - left
        start = 1

    if right > 0:
        middle = nr - right
    else:
        middle = nr

    for j in range(nc_actual):

        j_act = locs[j]
        temp = a[:, j_act]

        # must enter first for loop before subtraction can begin
        for i in range(start):
            if not_started:
                if i + right > 0:
                    not_started = False
                    n = 0
                    for k in range(max(0, i + left), i + right):
                        if temp[k] is not None:
                            n += 1
                else:
                    continue
            else:
                if temp[i + right - 1] is not None:
                    n += 1

            if n != 0:
                result[i, j] = n

        for i in range(start, middle):
            if temp[i + right - 1] is not None:
                n += 1
            if temp[i + left - 1] is not None:
                n -= 1
            result[i, j] = n

        if middle != nr:
            if temp[i + right] is not None:
                n += 1

        for i in range(middle, end):
            if temp[i + left - 1] is not None:
                n -= 1
            result[i, j] = n

    return result

def prod_int(ndarray[np.int64_t, ndim=2] a, ndarray[np.int64_t] locs,
            int left, int right, int min_window):
    cdef:
        Py_ssize_t i=0, j, k, j_act, start, middle, end, ct = 0
        Py_ssize_t nr = a.shape[0], nc = a.shape[1], nc_actual = len(locs)
        ndarray[np.int64_t, ndim=2] result = np.zeros((nr, nc_actual), dtype='int64')
        ndarray[np.int64_t] temp
        np.int64_t total = 1
        bint not_started = True

    if left < 0:
        start = -left + 1
        end = nr
    else:
        end = nr - left
        start = 1

    if right > 0:
        middle = nr - right
    else:
        middle = nr

    for j in range(nc_actual):

        j_act = locs[j]
        temp = a[:, j_act]

        # must enter first for loop before subtraction can begin
        for i in range(start):
            if not_started:
                if i + right > 0:
                    not_started = False
                    for k in range(max(0, i + left), i + right):
                        total *= temp[k]
                else:
                    continue
            else:
                total *= temp[i + right - 1]
            result[i, j] = total

        for i in range(start, middle):
            if temp[i + left - 1] == 0:
                total = 1
                for k in range(i + left, i + right):
                    total *= temp[k]
            else:
                total = total * temp[i + right - 1] / temp[i + left - 1]
            result[i, j] = total

        if middle != nr:
            total *= temp[i + right]

        for i in range(middle, end):
            if temp[i + left - 1] == 0:
                total = 1
                for k in range(i + left, nr):
                    total *= temp[k]
            else:
                total /= temp[i + left - 1]
            result[i, j] = total

    return result

def prod_float(ndarray[np.float64_t, ndim=2] a, ndarray[np.int64_t] locs,
            int left, int right, int min_window):
    cdef:
        Py_ssize_t i=0, j, k, j_act, start, middle, end, ct = 0
        Py_ssize_t nr = a.shape[0], nc = a.shape[1], nc_actual = len(locs)
        ndarray[np.float64_t, ndim=2] result = np.zeros((nr, nc_actual), dtype='float64')
        ndarray[np.float64_t] temp
        np.float64_t total = 1
        bint not_started = True

    if left < 0:
        start = -left + 1
        end = nr
    else:
        end = nr - left
        start = 1

    if right > 0:
        middle = nr - right
    else:
        middle = nr

    for j in range(nc_actual):

        j_act = locs[j]
        temp = a[:, j_act]

        # must enter first for loop before subtraction can begin
        for i in range(start):
            if not_started:
                if i + right > 0:
                    not_started = False
                    for k in range(max(0, i + left), i + right):
                        if not npy_isnan(temp[k]):
                            total *= temp[k]
                else:
                    continue
            else:
                if npy_isnan(temp[i + right - 1]):
                    total *= temp[i + right - 1]
            result[i, j] = total

        for i in range(start, middle):
            if temp[i + left - 1] == 0:
                total = 1
                for k in range(i + left, i + right):
                    if not npy_isnan(temp[k]):
                        total *= temp[k]
            else:
                if not npy_isnan(temp[i + right - 1]):
                    total = total * temp[i + right - 1]
                if not npy_isnan(temp[i + left - 1]):
                    total /= temp[i + left - 1]
            result[i, j] = total

        if middle != nr:
            if not npy_isnan(temp[i + right]):
                total *= temp[i + right]

        for i in range(middle, end):
            if temp[i + left - 1] == 0:
                total = 1
                for k in range(i + left, nr):
                    if not npy_isnan(temp[k]):
                        total *= temp[k]
            else:
                if not npy_isnan(temp[i + left - 1]):
                    total /= temp[i + left - 1]
            result[i, j] = total

    return result

def prod_bool(ndarray[np.uint8_t, ndim=2, cast=True] a, ndarray[np.int64_t] locs,
            int left, int right, int min_window):
    cdef:
        Py_ssize_t i=0, j, k, j_act, start, middle, end, ct = 0
        Py_ssize_t nr = a.shape[0], nc = a.shape[1], nc_actual = len(locs)
        ndarray[np.int64_t, ndim=2] result = np.zeros((nr, nc_actual), dtype='int64')
        ndarray[np.uint8_t, cast=True] temp
        np.int64_t total = 1
        bint not_started = True

    if left < 0:
        start = -left + 1
        end = nr
    else:
        end = nr - left
        start = 1

    if right > 0:
        middle = nr - right
    else:
        middle = nr

    for j in range(nc_actual):

        j_act = locs[j]
        temp = a[:, j_act]

        # must enter first for loop before subtraction can begin
        for i in range(start):
            if not_started:
                if i + right > 0:
                    not_started = False
                    for k in range(max(0, i + left), i + right):
                        total *= temp[k]
                else:
                    continue
            else:
                total *= temp[i + right - 1]
            result[i, j] = total

        for i in range(start, middle):
            if temp[i + left - 1] == 0:
                total = 1
                for k in range(i + left, i + right):
                    total *= temp[k]
            else:
                total = total * temp[i + right - 1] / temp[i + left - 1]
            result[i, j] = total

        if middle != nr:
            total *= temp[i + right]

        for i in range(middle, end):
            if temp[i + left - 1] == 0:
                total = 1
                for k in range(i + left, nr):
                    total *= temp[k]
            else:
                total /= temp[i + left - 1]
            result[i, j] = total

    return result

def median_int(ndarray[np.int64_t, ndim=2] a, ndarray[np.int64_t] locs,
               int left, int right, int min_window):
    cdef:
        Py_ssize_t i=0, j, k, j_act, l=0, n=0, n1 = right - left
        Py_ssize_t nr = a.shape[0], nc = a.shape[1], nc_actual = len(locs)
        Py_ssize_t first_n=0, middle_n=nr, last_n=0
        ndarray[np.float64_t, ndim=2] result = np.full((nr, nc_actual), nan, dtype='float64')
        ndarray[np.int64_t] data = np.empty(n1, dtype='int64')
        bint not_started = True

    if left < 0:
        start = -left + 1
        end = nr
    else:
        end = nr - left
        start = 1

    if right > 0:
        middle = nr - right
    else:
        middle = nr

    for j in range(nc_actual):
        j_act = locs[j]

        # must enter first for loop before subtraction can begin
        for i in range(start):
            if not_started:
                if i + right > 0:
                    not_started = False
                    for k in range(max(0, i + left), i + right):
                        data[n] = a[k, j_act]
                        n += 1
                else:
                    continue
            else:
                data[n] = a[i + right - 1, j_act]
                n += 1
            result[i, j] = bn.median(data[:n])

        for i in range(start, middle):
            l = 0
            for k in range(i + left, i + right):
                data[l] = a[k, j_act]
                l += 1
            result[i, j] = bn.median(data)

        for i in range(middle, end):
            l = 0
            for k in range(i + left, nr):
                data[l] = a[k, j_act]
                l += 1
            result[i, j] = bn.median(data[:l])

    return result

def median_float(ndarray[np.float64_t, ndim=2] a, ndarray[np.int64_t] locs,
            int left, int right, int min_window):
    cdef:
        Py_ssize_t i=0, j, k, j_act, l=0, n=0, n1 = right - left
        Py_ssize_t nr = a.shape[0], nc = a.shape[1], nc_actual = len(locs)
        Py_ssize_t first_n=0, middle_n=nr, last_n=0
        ndarray[np.float64_t, ndim=2] result = np.full((nr, nc_actual), nan, dtype='float64')
        ndarray[np.float64_t] data = np.full(right - left, nan, dtype='float64')
        bint not_started = True

    if left < 0:
        start = -left + 1
        end = nr
    else:
        end = nr - left
        start = 1

    if right > 0:
        middle = nr - right
    else:
        middle = nr

    for j in range(nc_actual):
        j_act = locs[j]

        # must enter first for loop before subtraction can begin
        for i in range(start):
            if not_started:
                if i + right > 0:
                    not_started = False
                    for k in range(max(0, i + left), i + right):
                        if not npy_isnan(a[k, j_act]):
                            data[n] = a[k, j_act]
                            n += 1
                else:
                    continue
            else:
                if not npy_isnan(a[i + right - 1, j_act]):
                    data[n] = a[i + right - 1, j_act]
                    n += 1
            if n > 0:
                result[i, j] = bn.median(data[:n])

        for i in range(start, middle):
            l = 0
            for k in range(i + left, i + right):
                if not npy_isnan(a[k, j_act]):
                    data[l] = a[k, j_act]
                    l += 1
            if l > 0:
                result[i, j] = bn.median(data[:l])

        for i in range(middle, end):
            l = 0
            for k in range(i + left, nr):
                if not npy_isnan(a[k, j_act]):
                    data[l] = a[k, j_act]
                    l += 1
            if l > 0:
                result[i, j] = bn.median(data[:l])

    return result

def median_bool(ndarray[np.uint8_t, ndim=2, cast=True] a, ndarray[np.int64_t] locs,
            int left, int right, int min_window):
    cdef:
        Py_ssize_t i=0, j, k, j_act, l=0, n=0, n1 = right - left
        Py_ssize_t nr = a.shape[0], nc = a.shape[1], nc_actual = len(locs)
        Py_ssize_t first_n=0, middle_n=nr, last_n=0
        ndarray[np.float64_t, ndim=2] result = np.full((nr, nc_actual), nan, dtype='float64')
        ndarray[np.uint8_t, cast=True] data = np.empty(n1, dtype='bool')
        bint not_started = True

    if left < 0:
        start = -left + 1
        end = nr
    else:
        end = nr - left
        start = 1

    if right > 0:
        middle = nr - right
    else:
        middle = nr

    for j in range(nc_actual):
        j_act = locs[j]

        # must enter first for loop before subtraction can begin
        for i in range(start):
            if not_started:
                if i + right > 0:
                    not_started = False
                    for k in range(max(0, i + left), i + right):
                        data[n] = a[k, j_act]
                        n += 1
                else:
                    continue
            else:
                data[n] = a[i + right - 1, j_act]
                n += 1
            result[i, j] = bn.median(data[:n])

        for i in range(start, middle):
            l = 0
            for k in range(i + left, i + right):
                data[l] = a[k, j_act]
                l += 1
            result[i, j] = bn.median(data)

        for i in range(middle, end):
            l = 0
            for k in range(i + left, nr):
                data[l] = a[k, j_act]
                l += 1
            result[i, j] = bn.median(data[:l])

    return result

def var_int(ndarray[np.int64_t, ndim=2] a, ndarray[np.int64_t] locs,
            int left, int right, int min_window, int ddof):
    cdef:
        Py_ssize_t i=0, j, k, j_act, start, middle, end, ct = 0, n, n1 = right - left
        Py_ssize_t nr = a.shape[0], nc = a.shape[1], nc_actual = len(locs)
        ndarray[np.float64_t, ndim=2] result = np.full((nr, nc_actual), nan, dtype='float64')
        ndarray[np.int64_t] temp
        np.float64_t ex = 0, ex2 = 0, total=0
        bint not_started = True

    if left < 0:
        start = -left + 1
        end = nr
    else:
        end = nr - left
        start = 1

    if right > 0:
        middle = nr - right
    else:
        middle = nr

    for j in range(nc_actual):
        j_act = locs[j]
        temp = a[:, j_act]

        # must enter first for loop before subtraction can begin
        for i in range(start):
            if not_started:
                if i + right > 0:
                    not_started = False
                    n = 0
                    for k in range(max(0, i + left), i + right):
                        ex += temp[k]
                        ex2 += temp[k] ** 2
                        n += 1
                else:
                    continue
            else:
                n += 1
                ex += temp[i + right - 1]
                ex2 += temp[i + right - 1] ** 2
            if n > ddof:
                result[i, j] = ex2 / (n - ddof) - ex ** 2 / (n * (n - ddof))

        if n1 > ddof:
            for i in range(start, middle):
                ex = ex + temp[i + right - 1] - temp[i + left - 1]
                ex2 = ex2 + temp[i + right - 1] ** 2 - (temp[i + left - 1] ** 2)
                result[i, j] = ex2 / (n1 - ddof) - ex ** 2 / (n1 * (n1 - ddof))

        if middle != nr:
            ex += temp[i + right]
            ex2 += temp[i + right] ** 2
            n1 += 1

        for i in range(middle, end):
            ex -= temp[i + left - 1]
            ex2 -= (temp[i + left - 1] ** 2)
            n1 -= 1
            if n1 > ddof:
                result[i, j] = ex2 / (n1 - ddof) - ex ** 2 / (n1 * (n1 - ddof))
    return result

def var_float(ndarray[np.float64_t, ndim=2] a, ndarray[np.int64_t] locs,
            int left, int right, int min_window, int ddof):
    cdef:
        Py_ssize_t i=0, j, k, j_act, start, middle, end, ct = 0, n, n1 = right - left
        Py_ssize_t nr = a.shape[0], nc = a.shape[1], nc_actual = len(locs)
        ndarray[np.float64_t, ndim=2] result = np.full((nr, nc_actual), nan, dtype='float64')
        ndarray[np.float64_t] temp
        np.float64_t ex = 0, ex2 = 0, total=0
        bint not_started = True

    if left < 0:
        start = -left + 1
        end = nr
    else:
        end = nr - left
        start = 1

    if right > 0:
        middle = nr - right
    else:
        middle = nr

    for j in range(nc_actual):
        j_act = locs[j]
        temp = a[:, j_act]

        # must enter first for loop before subtraction can begin
        for i in range(start):
            if not_started:
                if i + right > 0:
                    not_started = False
                    n = 0
                    for k in range(max(0, i + left), i + right):
                        if not npy_isnan(temp[k]):
                            ex += temp[k]
                            ex2 += temp[k] ** 2
                            n += 1
                else:
                    continue
            else:
                if not npy_isnan(temp[i + right - 1]):
                    ex += temp[i + right - 1]
                    ex2 += temp[i + right - 1] ** 2
                    n += 1
            if n > ddof:
                result[i, j] = ex2 / (n - ddof) - ex ** 2 / (n * (n - ddof))

        for i in range(start, middle):
            if not npy_isnan(temp[i + right - 1]):
                ex = ex + temp[i + right - 1]
                ex2 = ex2 + temp[i + right - 1] ** 2
                n += 1

            if not npy_isnan(temp[i + left - 1]):
                ex = ex - temp[i + left - 1]
                ex2 = ex2 - temp[i + left - 1] ** 2
                n -= 1
            if n > ddof:
                result[i, j] = ex2 / (n - ddof) - ex ** 2 / (n * (n - ddof))

        if middle != nr:
            if not npy_isnan(temp[i + right]):
                ex += temp[i + right]
                ex2 += temp[i + right] ** 2
                n += 1

        for i in range(middle, end):
            if not npy_isnan(temp[i + left - 1]):
                ex -= temp[i + left - 1]
                ex2 -= (temp[i + left - 1] ** 2)
                n -= 1
            if n > ddof:
                result[i, j] = ex2 / (n - ddof) - ex ** 2 / (n * (n - ddof))
    return result

def var_bool(ndarray[np.uint8_t, ndim=2, cast=True] a, ndarray[np.int64_t] locs,
            int left, int right, int min_window, int ddof):
    cdef:
        Py_ssize_t i=0, j, k, j_act, start, middle, end, ct = 0, n, n1 = right - left
        Py_ssize_t nr = a.shape[0], nc = a.shape[1], nc_actual = len(locs)
        ndarray[np.float64_t, ndim=2] result = np.full((nr, nc_actual), nan, dtype='float64')
        ndarray[np.uint8_t, cast=True] temp
        np.float64_t ex = 0, ex2 = 0, total=0
        bint not_started = True

    if left < 0:
        start = -left + 1
        end = nr
    else:
        end = nr - left
        start = 1

    if right > 0:
        middle = nr - right
    else:
        middle = nr

    for j in range(nc_actual):
        j_act = locs[j]
        temp = a[:, j_act]

        # must enter first for loop before subtraction can begin
        for i in range(start):
            if not_started:
                if i + right > 0:
                    not_started = False
                    n = 0
                    for k in range(max(0, i + left), i + right):
                        ex += temp[k]
                        ex2 += temp[k] ** 2
                        n += 1
                else:
                    continue
            else:
                n += 1
                ex += temp[i + right - 1]
                ex2 += temp[i + right - 1] ** 2
            if n > ddof:
                result[i, j] = ex2 / (n - ddof) - ex ** 2 / (n * (n - ddof))

        if n1 > ddof:
            for i in range(start, middle):
                ex = ex + temp[i + right - 1] - temp[i + left - 1]
                ex2 = ex2 + temp[i + right - 1] ** 2 - (temp[i + left - 1] ** 2)
                result[i, j] = ex2 / (n1 - ddof) - ex ** 2 / (n1 * (n1 - ddof))

        if middle != nr:
            ex += temp[i + right]
            ex2 += temp[i + right] ** 2
            n1 += 1

        for i in range(middle, end):
            ex -= temp[i + left - 1]
            ex2 -= (temp[i + left - 1] ** 2)
            n1 -= 1
            if n1 > ddof:
                result[i, j] = ex2 / (n1 - ddof) - ex ** 2 / (n1 * (n1 - ddof))
    return result


def std_int(ndarray[np.int64_t, ndim=2] a, ndarray[np.int64_t] locs,
            int left, int right, int min_window, int ddof):
    cdef:
        Py_ssize_t i=0, j, k, j_act, start, middle, end, ct = 0, n, n1 = right - left
        Py_ssize_t nr = a.shape[0], nc = a.shape[1], nc_actual = len(locs)
        ndarray[np.float64_t, ndim=2] result = np.full((nr, nc_actual), nan, dtype='float64')
        ndarray[np.int64_t] temp
        np.float64_t ex = 0, ex2 = 0, total=0
        bint not_started = True

    if left < 0:
        start = -left + 1
        end = nr
    else:
        end = nr - left
        start = 1

    if right > 0:
        middle = nr - right
    else:
        middle = nr

    for j in range(nc_actual):
        j_act = locs[j]
        temp = a[:, j_act]

        # must enter first for loop before subtraction can begin
        for i in range(start):
            if not_started:
                if i + right > 0:
                    not_started = False
                    n = 0
                    for k in range(max(0, i + left), i + right):
                        ex += temp[k]
                        ex2 += temp[k] ** 2
                        n += 1
                else:
                    continue
            else:
                n += 1
                ex += temp[i + right - 1]
                ex2 += temp[i + right - 1] ** 2
            if n > ddof:
                result[i, j] = np.sqrt(ex2 / (n - ddof) - ex ** 2 / (n * (n - ddof)))

        if n1 > ddof:
            for i in range(start, middle):
                ex = ex + temp[i + right - 1] - temp[i + left - 1]
                ex2 = ex2 + temp[i + right - 1] ** 2 - (temp[i + left - 1] ** 2)
                result[i, j] = np.sqrt(ex2 / (n1 - ddof) - ex ** 2 / (n1 * (n1 - ddof)))

        if middle != nr:
            ex += temp[i + right]
            ex2 += temp[i + right] ** 2
            n1 += 1

        for i in range(middle, end):
            ex -= temp[i + left - 1]
            ex2 -= (temp[i + left - 1] ** 2)
            n1 -= 1
            if n1 > ddof:
                result[i, j] = np.sqrt(ex2 / (n1 - ddof) - ex ** 2 / (n1 * (n1 - ddof)))
    return result

def std_float(ndarray[np.float64_t, ndim=2] a, ndarray[np.int64_t] locs,
            int left, int right, int min_window, int ddof):
    cdef:
        Py_ssize_t i=0, j, k, j_act, start, middle, end, ct = 0, n, n1 = right - left
        Py_ssize_t nr = a.shape[0], nc = a.shape[1], nc_actual = len(locs)
        ndarray[np.float64_t, ndim=2] result = np.full((nr, nc_actual), nan, dtype='float64')
        ndarray[np.float64_t] temp
        np.float64_t ex = 0, ex2 = 0, total=0
        bint not_started = True

    if left < 0:
        start = -left + 1
        end = nr
    else:
        end = nr - left
        start = 1

    if right > 0:
        middle = nr - right
    else:
        middle = nr

    for j in range(nc_actual):
        j_act = locs[j]
        temp = a[:, j_act]

        # must enter first for loop before subtraction can begin
        for i in range(start):
            if not_started:
                if i + right > 0:
                    not_started = False
                    n = 0
                    for k in range(max(0, i + left), i + right):
                        if not npy_isnan(temp[k]):
                            ex += temp[k]
                            ex2 += temp[k] ** 2
                            n += 1
                else:
                    continue
            else:
                if not npy_isnan(temp[i + right - 1]):
                    ex += temp[i + right - 1]
                    ex2 += temp[i + right - 1] ** 2
                    n += 1
            if n > ddof:
                result[i, j] = np.sqrt(ex2 / (n - ddof) - ex ** 2 / (n * (n - ddof)))

        for i in range(start, middle):
            if not npy_isnan(temp[i + right - 1]):
                ex = ex + temp[i + right - 1]
                ex2 = ex2 + temp[i + right - 1] ** 2
                n += 1

            if not npy_isnan(temp[i + left - 1]):
                ex = ex - temp[i + left - 1]
                ex2 = ex2 - temp[i + left - 1] ** 2
                n -= 1
            if n > ddof:
                result[i, j] = np.sqrt(ex2 / (n - ddof) - ex ** 2 / (n * (n - ddof)))

        if middle != nr:
            if not npy_isnan(temp[i + right]):
                ex += temp[i + right]
                ex2 += temp[i + right] ** 2
                n += 1

        for i in range(middle, end):
            if not npy_isnan(temp[i + left - 1]):
                ex -= temp[i + left - 1]
                ex2 -= (temp[i + left - 1] ** 2)
                n -= 1
            if n > ddof:
                result[i, j] = np.sqrt(ex2 / (n - ddof) - ex ** 2 / (n * (n - ddof)))
    return result

def std_bool(ndarray[np.uint8_t, ndim=2, cast=True] a, ndarray[np.int64_t] locs,
            int left, int right, int min_window, int ddof):
    cdef:
        Py_ssize_t i=0, j, k, j_act, start, middle, end, ct = 0, n, n1 = right - left
        Py_ssize_t nr = a.shape[0], nc = a.shape[1], nc_actual = len(locs)
        ndarray[np.float64_t, ndim=2] result = np.full((nr, nc_actual), nan, dtype='float64')
        ndarray[np.uint8_t, cast=True] temp
        np.float64_t ex = 0, ex2 = 0, total=0
        bint not_started = True

    if left < 0:
        start = -left + 1
        end = nr
    else:
        end = nr - left
        start = 1

    if right > 0:
        middle = nr - right
    else:
        middle = nr

    for j in range(nc_actual):
        j_act = locs[j]
        temp = a[:, j_act]

        # must enter first for loop before subtraction can begin
        for i in range(start):
            if not_started:
                if i + right > 0:
                    not_started = False
                    n = 0
                    for k in range(max(0, i + left), i + right):
                        ex += temp[k]
                        ex2 += temp[k] ** 2
                        n += 1
                else:
                    continue
            else:
                n += 1
                ex += temp[i + right - 1]
                ex2 += temp[i + right - 1] ** 2
            if n > ddof:
                result[i, j] = np.sqrt(ex2 / (n - ddof) - ex ** 2 / (n * (n - ddof)))

        if n1 > ddof:
            for i in range(start, middle):
                ex = ex + temp[i + right - 1] - temp[i + left - 1]
                ex2 = ex2 + temp[i + right - 1] ** 2 - (temp[i + left - 1] ** 2)
                result[i, j] = np.sqrt(ex2 / (n1 - ddof) - ex ** 2 / (n1 * (n1 - ddof)))

        if middle != nr:
            ex += temp[i + right]
            ex2 += temp[i + right] ** 2
            n1 += 1

        for i in range(middle, end):
            ex -= temp[i + left - 1]
            ex2 -= (temp[i + left - 1] ** 2)
            n1 -= 1
            if n1 > ddof:
                result[i, j] = np.sqrt(ex2 / (n1 - ddof) - ex ** 2 / (n1 * (n1 - ddof)))
    return result

def nunique_int(ndarray[np.int64_t, ndim=2] a, ndarray[np.int64_t] locs,
            int left, int right, int min_window):
    cdef:
        Py_ssize_t i=0, j, k, j_act, start, middle, end, ct = 0
        Py_ssize_t nr = a.shape[0], nc = a.shape[1], nc_actual = len(locs)
        ndarray[np.int64_t, ndim=2] result = np.zeros((nr, nc_actual), dtype='int64')
        ndarray[np.int64_t] temp
        bint not_started = True
        d = defaultdict(int)


    if left < 0:
        start = -left + 1
        end = nr
    else:
        end = nr - left
        start = 1

    if right > 0:
        middle = nr - right
    else:
        middle = nr

    for j in range(nc_actual):

        j_act = locs[j]
        temp = a[:, j_act]

        # must enter first for loop before subtraction can begin
        for i in range(start):
            if not_started:
                if i + right > 0:
                    not_started = False
                    for k in range(max(0, i + left), i + right):
                        d[temp[k]] += 1
                else:
                    continue
            else:
                d[temp[i + right - 1]] += 1
            result[i, j] = len(d)

        for i in range(start, middle):
            d[temp[i + left - 1]] -= 1
            if d[temp[i + left - 1]] == 0:
                d.pop(temp[i + left - 1])
            d[temp[i + right - 1]] += 1
            result[i, j] = len(d)

        if middle != nr:
            d[temp[i + right]] += 1

        for i in range(middle, end):
            d[temp[i + left - 1]] -= 1
            if d[temp[i + left - 1]] == 0:
                d.pop(temp[i + left - 1])
            result[i, j] = len(d)

    return result

def nunique_float(ndarray[np.float64_t, ndim=2] a, ndarray[np.int64_t] locs,
            int left, int right, int min_window):
    cdef:
        Py_ssize_t i=0, j, k, j_act, start, middle, end
        Py_ssize_t nr = a.shape[0], nc = a.shape[1], nc_actual = len(locs)
        ndarray[np.int64_t, ndim=2] result = np.zeros((nr, nc_actual), dtype='int64')
        ndarray[np.float64_t] temp
        bint not_started = True
        d = defaultdict(int)


    if left < 0:
        start = -left + 1
        end = nr
    else:
        end = nr - left
        start = 1

    if right > 0:
        middle = nr - right
    else:
        middle = nr

    for j in range(nc_actual):

        j_act = locs[j]
        temp = a[:, j_act]

        for i in range(start):
            if not_started:
                if i + right > 0:
                    not_started = False
                    for k in range(max(0, i + left), i + right):
                        if npy_isnan(temp[k]):
                            d[nan] += 1
                        else:
                            d[temp[k]] += 1
                else:
                    continue
            else:
                if npy_isnan(temp[i + right - 1]):
                    d[nan] += 1
                else:
                    d[temp[i + right - 1]] += 1
            result[i, j] = len(d)

        for i in range(start, middle):
            if npy_isnan(temp[i + left - 1]):
                d[nan] -= 1
                if d[nan] == 0:
                    d.pop(nan)
            else:
                d[temp[i + left - 1]] -= 1
                if d[temp[i + left - 1]] == 0:
                    d.pop(temp[i + left - 1])

            if npy_isnan(temp[i + right - 1]):
                d[nan] += 1
            else:
                d[temp[i + right - 1]] += 1
            result[i, j] = len(d)

        if middle != nr:
            if npy_isnan(temp[i + right]):
                d[nan] += 1
            else:
                d[temp[i + right]] += 1

        for i in range(middle, end):
            if npy_isnan(temp[i + left - 1]):
                d[nan] -= 1
                if d[nan] == 0:
                    d.pop(nan)
            else:
                d[temp[i + left - 1]] -= 1
                if d[temp[i + left - 1]] == 0:
                    d.pop(temp[i + left - 1])

            result[i, j] = len(d)

    return result

def nunique_bool(ndarray[np.uint8_t, ndim=2, cast=True] a, ndarray[np.int64_t] locs,
            int left, int right, int min_window):
    cdef:
        Py_ssize_t i=0, j, k, j_act, start, middle, end, ct = 0
        Py_ssize_t nr = a.shape[0], nc = a.shape[1], nc_actual = len(locs)
        ndarray[np.int64_t, ndim=2] result = np.zeros((nr, nc_actual), dtype='int64')
        ndarray[np.int64_t] temp
        bint not_started = True
        d = defaultdict(int)


    if left < 0:
        start = -left + 1
        end = nr
    else:
        end = nr - left
        start = 1

    if right > 0:
        middle = nr - right
    else:
        middle = nr

    for j in range(nc_actual):

        j_act = locs[j]
        temp = a[:, j_act]

        # must enter first for loop before subtraction can begin
        for i in range(start):
            if not_started:
                if i + right > 0:
                    not_started = False
                    for k in range(max(0, i + left), i + right):
                        d[temp[k]] += 1
                else:
                    continue
            else:
                d[temp[i + right - 1]] += 1
            result[i, j] = len(d)

        for i in range(start, middle):
            d[temp[i + left - 1]] -= 1
            if d[temp[i + left - 1]] == 0:
                d.pop(temp[i + left - 1])
            d[temp[i + right - 1]] += 1
            result[i, j] = len(d)

        if middle != nr:
            d[temp[i + right]] += 1

        for i in range(middle, end):
            d[temp[i + left - 1]] -= 1
            if d[temp[i + left - 1]] == 0:
                d.pop(temp[i + left - 1])
            result[i, j] = len(d)

    return result

def nunique_str(ndarray[object, ndim=2] a, ndarray[np.int64_t] locs,
            int left, int right, int min_window):
    cdef:
        Py_ssize_t i=0, j, k, j_act, start, middle, end, ct = 0
        Py_ssize_t nr = a.shape[0], nc = a.shape[1], nc_actual = len(locs)
        ndarray[np.int64_t, ndim=2] result = np.zeros((nr, nc_actual), dtype='int64')
        ndarray[object] temp
        bint not_started = True
        d = defaultdict(int)

    if left < 0:
        start = -left + 1
        end = nr
    else:
        end = nr - left
        start = 1

    if right > 0:
        middle = nr - right
    else:
        middle = nr

    for j in range(nc_actual):

        j_act = locs[j]
        temp = a[:, j_act]

        # must enter first for loop before subtraction can begin
        for i in range(start):
            if not_started:
                if i + right > 0:
                    not_started = False
                    for k in range(max(0, i + left), i + right):
                        d[temp[k]] += 1
                else:
                    continue
            else:
                d[temp[i + right - 1]] += 1
            result[i, j] = len(d)

        for i in range(start, middle):
            d[temp[i + left - 1]] -= 1
            if d[temp[i + left - 1]] == 0:
                d.pop(temp[i + left - 1])
            d[temp[i + right - 1]] += 1
            result[i, j] = len(d)

        if middle != nr:
            d[temp[i + right]] += 1

        for i in range(middle, end):
            d[temp[i + left - 1]] -= 1
            if d[temp[i + left - 1]] == 0:
                d.pop(temp[i + left - 1])
            result[i, j] = len(d)

    return result

def mode_int(ndarray[np.int64_t, ndim=2] a, ndarray[np.int64_t] locs,
            int left, int right, int min_window):
    cdef:
        Py_ssize_t i=0, j, k, j_act, start, middle, end, ct = 0, ct2=0, prev_ct=0, cur_ct=0
        Py_ssize_t nr = a.shape[0], nc = a.shape[1], nc_actual = len(locs)
        ndarray[np.int64_t, ndim=2] result = np.zeros((nr, nc_actual), dtype='int64')
        ndarray[np.int64_t] temp
        bint not_started = True
        d = defaultdict(int)
        np.int64_t val

    if left < 0:
        start = -left + 1
        end = nr
    else:
        end = nr - left
        start = 1

    if right > 0:
        middle = nr - right
    else:
        middle = nr

    for j in range(nc_actual):

        j_act = locs[j]
        temp = a[:, j_act]

        # must enter first for loop before subtraction can begin
        for i in range(start):
            if not_started:
                if i + right > 0:
                    not_started = False
                    for k in range(max(0, i + left), i + right):
                        d[temp[k]] += 1
                        if d[temp[k]] > ct:
                            val = temp[k]
                            ct = d[temp[k]]
                else:
                    continue
            else:
                d[temp[i + right - 1]] += 1
                if d[temp[i + right - 1]] > ct:
                    val = temp[i + right - 1]
                    ct = d[temp[i + right - 1]]

            result[i, j] = val
            prev_ct = ct

        for i in range(start, middle):
            d[temp[i + left - 1]] -= 1
            if d[temp[i + left - 1]] == 0:
                d.pop(temp[i + left - 1])
            d[temp[i + right - 1]] += 1

            ct = d.get(val, 0)
            if ct == prev_ct:
                if d[temp[i + right - 1]] > ct:
                    val = temp[i + right - 1]
                    ct += 1
            elif ct < prev_ct:
                if ct == 0 and d[temp[i + right - 1]] == 1:
                    val = temp[i + left]
                    ct = d[val]
                else:
                    ct = 0
                    for val2, ct2 in d.items():
                        if ct2 > ct:
                            ct = ct2
                            val = val2

            result[i, j] = val
            prev_ct = ct

        if middle != nr:
            d[temp[i + right]] += 1

        for i in range(middle, end):
            d[temp[i + left - 1]] -= 1
            if d[temp[i + left - 1]] == 0:
                d.pop(temp[i + left - 1])

            if val in d:
                ct = d[val]
            else:
                ct = 0

            for val2, ct2 in d.items():
                if ct2 > ct:
                    val = val2
                    ct = ct2

            result[i, j] = val

    return result

def mode_float(ndarray[np.float64_t, ndim=2] a, ndarray[np.int64_t] locs,
            int left, int right, int min_window):

    cdef:
        Py_ssize_t i=0, j, k, j_act, start, middle, end, ct = 0, ct2=0, prev_ct=0, cur_ct=0
        Py_ssize_t nr = a.shape[0], nc = a.shape[1], nc_actual = len(locs)
        ndarray[np.float64_t, ndim=2] result = np.full((nr, nc_actual), nan, dtype='float64')
        ndarray[np.float64_t] temp
        bint not_started = True
        d = defaultdict(int)
        np.float64_t val

    if left < 0:
        start = -left + 1
        end = nr
    else:
        end = nr - left
        start = 1

    if right > 0:
        middle = nr - right
    else:
        middle = nr

    for j in range(nc_actual):

        j_act = locs[j]
        temp = a[:, j_act]

        # must enter first for loop before subtraction can begin
        for i in range(start):
            if not_started:
                if i + right > 0:
                    not_started = False
                    for k in range(max(0, i + left), i + right):
                        d[temp[k]] += 1
                        if d[temp[k]] > ct:
                            val = temp[k]
                            ct = d[temp[k]]
                else:
                    continue
            else:
                d[temp[i + right - 1]] += 1
                if d[temp[i + right - 1]] > ct:
                    val = temp[i + right - 1]
                    ct = d[temp[i + right - 1]]

            result[i, j] = val
            prev_ct = ct

        for i in range(start, middle):
            d[temp[i + left - 1]] -= 1
            if d[temp[i + left - 1]] == 0:
                d.pop(temp[i + left - 1])
            d[temp[i + right - 1]] += 1

            ct = d.get(val, 0)
            if ct == prev_ct:
                if d[temp[i + right - 1]] > ct:
                    val = temp[i + right - 1]
                    ct += 1
            elif ct < prev_ct:
                if ct == 0 and d[temp[i + right - 1]] == 1:
                    val = temp[i + left]
                    ct = d[val]
                else:
                    ct = 0
                    for val2, ct2 in d.items():
                        if ct2 > ct:
                            ct = ct2
                            val = val2

            result[i, j] = val
            prev_ct = ct

        if middle != nr:
            d[temp[i + right]] += 1

        for i in range(middle, end):
            d[temp[i + left - 1]] -= 1
            if d[temp[i + left - 1]] == 0:
                d.pop(temp[i + left - 1])

            if val in d:
                ct = d[val]
            else:
                ct = 0

            for val2, ct2 in d.items():
                if ct2 > ct:
                    val = val2
                    ct = ct2

            result[i, j] = val

    return result

def mode_bool(ndarray[np.uint8_t, ndim=2, cast=True] a, ndarray[np.int64_t] locs,
            int left, int right, int min_window):
    cdef:
        Py_ssize_t i=0, j, k, j_act, start, middle, end, ct = 0, ct2=0, prev_ct=0, cur_ct=0
        Py_ssize_t nr = a.shape[0], nc = a.shape[1], nc_actual = len(locs)
        ndarray[np.uint8_t, ndim=2] result = np.zeros((nr, nc_actual), dtype='bool')
        ndarray[np.uint8_t] temp
        bint not_started = True
        d = defaultdict(int)
        bint val

    if left < 0:
        start = -left + 1
        end = nr
    else:
        end = nr - left
        start = 1

    if right > 0:
        middle = nr - right
    else:
        middle = nr

    for j in range(nc_actual):

        j_act = locs[j]
        temp = a[:, j_act]

        # must enter first for loop before subtraction can begin
        for i in range(start):
            if not_started:
                if i + right > 0:
                    not_started = False
                    for k in range(max(0, i + left), i + right):
                        d[temp[k]] += 1
                        if d[temp[k]] > ct:
                            val = temp[k]
                            ct = d[temp[k]]
                else:
                    continue
            else:
                d[temp[i + right - 1]] += 1
                if d[temp[i + right - 1]] > ct:
                    val = temp[i + right - 1]
                    ct = d[temp[i + right - 1]]

            result[i, j] = val
            prev_ct = ct

        for i in range(start, middle):
            d[temp[i + left - 1]] -= 1
            if d[temp[i + left - 1]] == 0:
                d.pop(temp[i + left - 1])
            d[temp[i + right - 1]] += 1

            ct = d.get(val, 0)
            if ct == prev_ct:
                if d[temp[i + right - 1]] > ct:
                    val = temp[i + right - 1]
                    ct += 1
            elif ct < prev_ct:
                if ct == 0 and d[temp[i + right - 1]] == 1:
                    val = temp[i + left]
                    ct = d[val]
                else:
                    ct = 0
                    for val2, ct2 in d.items():
                        if ct2 > ct:
                            ct = ct2
                            val = val2

            result[i, j] = val
            prev_ct = ct

        if middle != nr:
            d[temp[i + right]] += 1

        for i in range(middle, end):
            d[temp[i + left - 1]] -= 1
            if d[temp[i + left - 1]] == 0:
                d.pop(temp[i + left - 1])

            if val in d:
                ct = d[val]
            else:
                ct = 0

            for val2, ct2 in d.items():
                if ct2 > ct:
                    val = val2
                    ct = ct2

            result[i, j] = val

    return result

def mode_str(ndarray[object, ndim=2] a, ndarray[np.int64_t] locs,
            int left, int right, int min_window):
    cdef:
        Py_ssize_t i=0, j, k, j_act, start, middle, end, ct = 0, ct2=0, prev_ct=0, cur_ct=0
        Py_ssize_t nr = a.shape[0], nc = a.shape[1], nc_actual = len(locs)
        ndarray[object, ndim=2] result = np.empty((nr, nc_actual), dtype='O')
        ndarray[object] temp
        bint not_started = True
        d = defaultdict(int)
        object val

    if left < 0:
        start = -left + 1
        end = nr
    else:
        end = nr - left
        start = 1

    if right > 0:
        middle = nr - right
    else:
        middle = nr

    for j in range(nc_actual):

        j_act = locs[j]
        temp = a[:, j_act]

        # must enter first for loop before subtraction can begin
        for i in range(start):
            if not_started:
                if i + right > 0:
                    not_started = False
                    for k in range(max(0, i + left), i + right):
                        d[temp[k]] += 1
                        if d[temp[k]] > ct:
                            val = temp[k]
                            ct = d[temp[k]]
                else:
                    continue
            else:
                d[temp[i + right - 1]] += 1
                if d[temp[i + right - 1]] > ct:
                    val = temp[i + right - 1]
                    ct = d[temp[i + right - 1]]

            result[i, j] = val
            prev_ct = ct

        for i in range(start, middle):
            d[temp[i + left - 1]] -= 1
            if d[temp[i + left - 1]] == 0:
                d.pop(temp[i + left - 1])
            d[temp[i + right - 1]] += 1

            ct = d.get(val, 0)
            if ct == prev_ct:
                if d[temp[i + right - 1]] > ct:
                    val = temp[i + right - 1]
                    ct += 1
            elif ct < prev_ct:
                if ct == 0 and d[temp[i + right - 1]] == 1:
                    val = temp[i + left]
                    ct = d[val]
                else:
                    ct = 0
                    for val2, ct2 in d.items():
                        if ct2 > ct:
                            ct = ct2
                            val = val2

            result[i, j] = val
            prev_ct = ct

        if middle != nr:
            d[temp[i + right]] += 1

        for i in range(middle, end):
            d[temp[i + left - 1]] -= 1
            if d[temp[i + left - 1]] == 0:
                d.pop(temp[i + left - 1])

            if val in d:
                ct = d[val]
            else:
                ct = 0

            for val2, ct2 in d.items():
                if ct2 > ct:
                    val = val2
                    ct = ct2

            result[i, j] = val

    return result