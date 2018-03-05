import numpy as np
from numpy import nan
from libc.math cimport isnan, sqrt
from numpy cimport ndarray
cimport numpy as np


cdef create_days_to_year():
    cdef long i, j, idx = 0
    cdef ndarray[np.int64_t] days_to_year = np.empty(366 * 800, dtype='int64')
    cdef long day_nano = 10 ** 9 * 86400

    for i in range(800):
        if i % 4 != 0:
            for j in range(365):
                days_to_year[j + idx] = i + 1600
            idx += 365
        elif i % 100 != 0:
            for j in range(366):
                days_to_year[j + idx] = i + 1600
            idx += 366
        elif i % 400 == 0:
            for j in range(366):
                days_to_year[j + idx] = i + 1600
            idx += 366
        else:
            for j in range(365):
                days_to_year[j + idx] = i + 1600
            idx += 365
    return days_to_year

def create_days_to_month():
    cdef long i, j, idx = 0
    cdef ndarray[np.int64_t] months365 = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], dtype='int64')
    cdef ndarray[np.int64_t] months366 = np.array([31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], dtype='int64')
    cdef ndarray[np.int64_t] days_to_month = np.empty(366 * 800, dtype='int64')

    months365 = np.repeat(np.arange(12), months365)
    months366 = np.repeat(np.arange(12), months366)

    for i in range(800):
        if i % 4 != 0:
            for j in range(365):
                days_to_month[idx + j] = months365[j]
            idx += 365
        elif i % 100 != 0:
            for j in range(366):
                days_to_month[idx + j] = months366[j]
            idx += 366
        elif i % 400 == 0:
            for j in range(366):
                days_to_month[idx + j] = months366[j]
            idx += 366
        else:
            for j in range(365):
                days_to_month[idx + j] = months365[j]
            idx += 365
    return days_to_month

def create_days_to_day():
    cdef long i, j, idx = 0
    cdef ndarray[np.int64_t] days365 = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], dtype='int64')
    cdef ndarray[np.int64_t] days366 = np.array([31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], dtype='int64')
    cdef ndarray[np.int64_t] days_to_day = np.empty(366 * 800, dtype='int64')

    days365 = np.concatenate([np.arange(1, val + 1) for val in days365])
    days366 = np.concatenate([np.arange(1, val + 1) for val in days366])

    for i in range(800):
        if i % 4 != 0:
            for j in range(365):
                days_to_day[idx + j] = days365[j]
            idx += 365
        elif i % 100 != 0:
            for j in range(366):
                days_to_day[idx + j] = days366[j]
            idx += 366
        elif i % 400 == 0:
            for j in range(366):
                days_to_day[idx + j] = days366[j]
            idx += 366
        else:
            for j in range(365):
                days_to_day[idx + j] = days365[j]
            idx += 365
    return days_to_day

def create_days_to_year_day():
    cdef long i, j, idx = 0
    cdef ndarray[np.int64_t] days_to_day = np.empty(366 * 800, dtype='int64')

    for i in range(800):
        if i % 4 != 0:
            for j in range(365):
                days_to_day[idx + j] = j + 1
            idx += 365
        elif i % 100 != 0:
            for j in range(366):
                days_to_day[idx + j] = j + 1
            idx += 366
        elif i % 400 == 0:
            for j in range(366):
                days_to_day[idx + j] = j + 1
            idx += 366
        else:
            for j in range(365):
                days_to_day[idx + j] = j + 1
            idx += 365
    return days_to_day

def create_days_in_month():
    cdef long i, j, idx = 0
    cdef ndarray[np.int64_t] months365 = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], dtype='int64')
    cdef ndarray[np.int64_t] months366 = np.array([31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], dtype='int64')
    cdef ndarray[np.int64_t] days_in_month = np.empty(366 * 800, dtype='int64')

    months365 = np.repeat(months365, months365)
    months366 = np.repeat(months366, months366)

    for i in range(800):
        if i % 4 != 0:
            for j in range(365):
                days_in_month[idx + j] = months365[j]
            idx += 365
        elif i % 100 != 0:
            for j in range(366):
                days_in_month[idx + j] = months366[j]
            idx += 366
        elif i % 400 == 0:
            for j in range(366):
                days_in_month[idx + j] = months366[j]
            idx += 366
        else:
            for j in range(365):
                days_in_month[idx + j] = months365[j]
            idx += 365
    return days_in_month

def create_days_to_leap_year():
    cdef long i, j, idx = 0
    cdef ndarray[np.uint8_t, cast=True] days_to_leap = np.zeros(366 * 800, dtype='bool')

    for i in range(800):
        if (i % 4 == 0) and (i % 100 != 0 or i % 400 == 0):
            for j in range(366):
                days_to_leap[idx + j] = True
            idx += 366
        else:
            idx += 365

    return days_to_leap

def create_days_to_quarter_start():
    cdef long i, j, idx = 0
    cdef ndarray[np.uint8_t, cast=True] days_to_day = np.zeros(366 * 800, dtype='bool')

    for i in range(800):
        if i % 4 != 0:
            days_to_day[idx] = True
            days_to_day[idx + 90] = True
            days_to_day[idx + 181] = True
            days_to_day[idx + 273] = True
            idx += 365
        elif i % 100 != 0:
            days_to_day[idx] = True
            days_to_day[idx + 91] = True
            days_to_day[idx + 182] = True
            days_to_day[idx + 274] = True
            idx += 366
        elif i % 400 == 0:
            days_to_day[idx] = True
            days_to_day[idx + 91] = True
            days_to_day[idx + 182] = True
            days_to_day[idx + 274] = True
            idx += 366
        else:
            days_to_day[idx] = True
            days_to_day[idx + 90] = True
            days_to_day[idx + 181] = True
            days_to_day[idx + 273] = True
            idx += 365
    return days_to_day


def create_days_to_quarter_end():
    cdef long i, j, idx = 0
    cdef ndarray[np.uint8_t, cast=True] days_to_day = np.zeros(366 * 800, dtype='bool')

    for i in range(800):
        if i % 4 != 0:
            days_to_day[idx + 89] = True
            days_to_day[idx + 180] = True
            days_to_day[idx + 272] = True
            days_to_day[idx + 364] = True
            idx += 365
        elif i % 100 != 0:
            days_to_day[idx + 90] = True
            days_to_day[idx + 181] = True
            days_to_day[idx + 273] = True
            days_to_day[idx + 365] = True
            idx += 366
        elif i % 400 == 0:
            days_to_day[idx + 90] = True
            days_to_day[idx + 181] = True
            days_to_day[idx + 273] = True
            days_to_day[idx + 365] = True
            idx += 366
        else:
            days_to_day[idx + 89] = True
            days_to_day[idx + 180] = True
            days_to_day[idx + 272] = True
            days_to_day[idx + 364] = True
            idx += 365
    return days_to_day

def create_days_to_year_end():
    cdef long i, j, idx = 0
    cdef ndarray[np.uint8_t, cast=True] days_to_day = np.zeros(366 * 800, dtype='bool')

    for i in range(800):
        if i % 4 != 0:
            days_to_day[idx + 364] = True
            idx += 365
        elif i % 100 != 0:
            days_to_day[idx + 365] = True
            idx += 366
        elif i % 400 == 0:
            days_to_day[idx + 365] = True
            idx += 366
        else:
            days_to_day[idx + 364] = True
            idx += 365
    return days_to_day

def create_days_to_year_start():
    cdef long i, j, idx = 0
    cdef ndarray[np.uint8_t, cast=True] days_to_day = np.zeros(366 * 800, dtype='bool')

    for i in range(800):
        if i % 4 != 0:
            days_to_day[idx] = True
            idx += 365
        elif i % 100 != 0:
            days_to_day[idx] = True
            idx += 366
        elif i % 400 == 0:
            days_to_day[idx] = True
            idx += 366
        else:
            days_to_day[idx] = True
            idx += 365
    return days_to_day

def weekday_name(ndarray[np.int64_t, ndim=2] a):
    cdef int i, j
    cdef int nr = a.shape[0]
    cdef int nc = a.shape[1]
    cdef np.int64_t NAT = np.datetime64('nat').astype('int64')
    cdef ndarray[object, ndim=2] result = np.empty((nr, nc), dtype='O')
    cdef ndarray[object] day_count = np.array(['Thursday', 'Friday', 'Saturday', 'Sunday',
                                               'Monday', 'Tuesday', 'Wednesday'], 'O')

    for j in range(nc):
        for i in range(nr):
            if a[i, j] == NAT:
                result[i, j] = None
            else:
                result[i, j] = day_count[a[i, j] % 7]
    return result

def day_of_week(ndarray[np.int64_t, ndim=2] a):
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
                result[i, j] = (a[i, j] / year_nanos + 3) % 7
    return result

def month(ndarray[np.int64_t, ndim=2] a):
    cdef int i, j
    cdef int nr = a.shape[0]
    cdef int nc = a.shape[1]
    cdef ndarray[np.float64_t, ndim=2] result = np.empty((nr, nc), dtype='float64')
    cdef np.int64_t NAT = np.datetime64('nat').astype('int64')

    for j in range(nc):
        for i in range(nr):
            if a[i, j] == NAT:
                result[i, j] = nan
            else:
                result[i, j] = a[i, j] % 12 + 1
    return result

def month2(ndarray[np.int64_t, ndim=2] a):
    cdef int i, j
    cdef int nr = a.shape[0]
    cdef int nc = a.shape[1]
    cdef ndarray[np.float64_t, ndim=2] result = np.empty((nr, nc), dtype='float64')
    cdef np.int64_t NAT = np.datetime64('nat').astype('int64')
    cdef ndarray[np.int64_t] days_to_month
    cdef int days_since_1600 = 135140
    cdef long year_nanos = 10 ** 9 * 86400

    days_to_month = create_days_to_month()

    for j in range(nc):
        for i in range(nr):
            if a[i, j] == NAT:
                result[i, j] = nan
            else:
                result[i, j] = days_to_month[<long> (a[i, j] / year_nanos) + days_since_1600]
    return result

def year(ndarray[np.int64_t, ndim=2] a):
    cdef int i, j
    cdef int nr = a.shape[0]
    cdef int nc = a.shape[1]
    cdef ndarray[np.float64_t, ndim=2] result = np.empty((nr, nc), dtype='float64')
    cdef np.int64_t NAT = np.datetime64('nat').astype('int64')
    cdef ndarray[np.int64_t] days_to_year
    cdef int days_since_1600 = 135140
    cdef long year_nanos = 10 ** 9 * 86400

    days_to_year = create_days_to_year()

    for j in range(nc):
        for i in range(nr):
            if a[i, j] == NAT:
                result[i, j] = nan
            else:
                result[i, j] = days_to_year[<long> (a[i, j] / year_nanos) + days_since_1600]
    return result

def day(ndarray[np.int64_t, ndim=2] a):
    cdef int i, j
    cdef int nr = a.shape[0]
    cdef int nc = a.shape[1]
    cdef ndarray[np.float64_t, ndim=2] result = np.empty((nr, nc), dtype='float64')
    cdef np.int64_t NAT = np.datetime64('nat').astype('int64')
    cdef ndarray[np.int64_t] days_to_year
    cdef int days_since_1600 = 135140
    cdef long year_nanos = 10 ** 9 * 86400

    days_to_year = create_days_to_day()

    for j in range(nc):
        for i in range(nr):
            if a[i, j] == NAT:
                result[i, j] = nan
            else:
                result[i, j] = days_to_year[<long> (a[i, j] / year_nanos) + days_since_1600]
    return result

def day_of_year(ndarray[np.int64_t, ndim=2] a):
    cdef int i, j
    cdef int nr = a.shape[0]
    cdef int nc = a.shape[1]
    cdef ndarray[np.float64_t, ndim=2] result = np.empty((nr, nc), dtype='float64')
    cdef np.int64_t NAT = np.datetime64('nat').astype('int64')
    cdef ndarray[np.int64_t] days_to_year_day
    cdef int days_since_1600 = 135140
    cdef long year_nanos = 10 ** 9 * 86400

    days_to_year_day = create_days_to_year_day()

    for j in range(nc):
        for i in range(nr):
            if a[i, j] == NAT:
                result[i, j] = nan
            else:
                result[i, j] = days_to_year_day[<long> (a[i, j] / year_nanos) + days_since_1600]
    return result


def days_in_month(ndarray[np.int64_t, ndim=2] a):
    cdef int i, j
    cdef int nr = a.shape[0]
    cdef int nc = a.shape[1]
    cdef ndarray[np.float64_t, ndim=2] result = np.empty((nr, nc), dtype='float64')
    cdef np.int64_t NAT = np.datetime64('nat').astype('int64')
    cdef ndarray[np.int64_t] days_to_days_in_month
    cdef int days_since_1600 = 135140
    cdef long year_nanos = 10 ** 9 * 86400

    days_to_days_in_month = create_days_in_month()

    for j in range(nc):
        for i in range(nr):
            if a[i, j] == NAT:
                result[i, j] = nan
            else:
                result[i, j] = days_to_days_in_month[<long> (a[i, j] / year_nanos) + days_since_1600]
    return result

def is_leap_year(ndarray[np.int64_t, ndim=2] a):
    cdef int i, j
    cdef int nr = a.shape[0]
    cdef int nc = a.shape[1]
    cdef ndarray[np.uint8_t, cast=True, ndim=2] result = np.empty((nr, nc), dtype='bool')
    cdef np.int64_t NAT = np.datetime64('nat').astype('int64')
    cdef ndarray[np.uint8_t, cast=True] days_to_is_leap_year
    cdef int days_since_1600 = 135140
    cdef long year_nanos = 10 ** 9 * 86400

    days_to_is_leap_year = create_days_to_leap_year()

    for j in range(nc):
        for i in range(nr):
            if a[i, j] == NAT:
                result[i, j] = False
            else:
                result[i, j] = days_to_is_leap_year[<long> (a[i, j] / year_nanos) + days_since_1600]
    return result

def is_quarter_start(ndarray[np.int64_t, ndim=2] a):
    cdef int i, j
    cdef int nr = a.shape[0]
    cdef int nc = a.shape[1]
    cdef ndarray[np.uint8_t, cast=True, ndim=2] result = np.empty((nr, nc), dtype='bool')
    cdef np.int64_t NAT = np.datetime64('nat').astype('int64')
    cdef ndarray[np.uint8_t, cast=True] days_to_is_quarter_start
    cdef int days_since_1600 = 135140
    cdef long year_nanos = 10 ** 9 * 86400

    days_to_is_quarter_start = create_days_to_quarter_start()

    for j in range(nc):
        for i in range(nr):
            if a[i, j] == NAT:
                result[i, j] = False
            else:
                result[i, j] = days_to_is_quarter_start[<long> (a[i, j] / year_nanos) + days_since_1600]
    return result

def is_quarter_end(ndarray[np.int64_t, ndim=2] a):
    cdef int i, j
    cdef int nr = a.shape[0]
    cdef int nc = a.shape[1]
    cdef ndarray[np.uint8_t, cast=True, ndim=2] result = np.empty((nr, nc), dtype='bool')
    cdef np.int64_t NAT = np.datetime64('nat').astype('int64')
    cdef ndarray[np.uint8_t, cast=True] days_to_is_quarter_end
    cdef int days_since_1600 = 135140
    cdef long year_nanos = 10 ** 9 * 86400

    days_to_is_quarter_end = create_days_to_quarter_end()

    for j in range(nc):
        for i in range(nr):
            if a[i, j] == NAT:
                result[i, j] = False
            else:
                result[i, j] = days_to_is_quarter_end[<long> (a[i, j] / year_nanos) + days_since_1600]
    return result


def is_year_end(ndarray[np.int64_t, ndim=2] a):
    cdef int i, j
    cdef int nr = a.shape[0]
    cdef int nc = a.shape[1]
    cdef ndarray[np.uint8_t, cast=True, ndim=2] result = np.empty((nr, nc), dtype='bool')
    cdef np.int64_t NAT = np.datetime64('nat').astype('int64')
    cdef ndarray[np.uint8_t, cast=True] days_to_is_year_end
    cdef int days_since_1600 = 135140
    cdef long year_nanos = 10 ** 9 * 86400

    days_to_is_year_end = create_days_to_year_end()

    for j in range(nc):
        for i in range(nr):
            if a[i, j] == NAT:
                result[i, j] = False
            else:
                result[i, j] = days_to_is_year_end[<long> (a[i, j] / year_nanos) + days_since_1600]
    return result

def is_year_start(ndarray[np.int64_t, ndim=2] a):
    cdef int i, j
    cdef int nr = a.shape[0]
    cdef int nc = a.shape[1]
    cdef ndarray[np.uint8_t, cast=True, ndim=2] result = np.empty((nr, nc), dtype='bool')
    cdef np.int64_t NAT = np.datetime64('nat').astype('int64')
    cdef ndarray[np.uint8_t, cast=True] days_to_is_year_start
    cdef int days_since_1600 = 135140
    cdef long year_nanos = 10 ** 9 * 86400

    days_to_is_year_start = create_days_to_year_start()

    for j in range(nc):
        for i in range(nr):
            if a[i, j] == NAT:
                result[i, j] = False
            else:
                result[i, j] = days_to_is_year_start[<long> (a[i, j] / year_nanos) + days_since_1600]
    return result
