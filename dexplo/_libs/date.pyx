import numpy as np
from numpy import nan
from libc.math cimport ceil, round
from numpy cimport ndarray
cimport numpy as np
import cython
import datetime

MONTHS365 = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], dtype='int64')
MONTHS366 = np.array([31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], dtype='int64')

cdef:
    long YEAR_NANOS = 10 ** 9 * 86400
    np.int64_t NAT = np.datetime64('nat').astype('int64')
    int DAYS_SINCE_1600 = 135140

cdef create_days_to_year():
    cdef:
        Py_ssize_t i, j
        long idx = 0, day_nano = 10 ** 9 * 86400
        ndarray[np.int64_t] days_to_year = np.empty(366 * 800, dtype='int64')

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
    cdef:
        Py_ssize_t i, j
        long idx = 0
        ndarray[np.int64_t] days_to_month = np.empty(366 * 800, dtype='int64')
        ndarray[np.int64_t] months365, months366

    months365 = np.repeat(np.arange(12), MONTHS365)
    months366 = np.repeat(np.arange(12), MONTHS366)

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
    cdef:
        Py_ssize_t i, j
        long idx = 0
        ndarray[np.int64_t] days_to_day = np.empty(366 * 800, dtype='int64')
        ndarray[np.int64_t] months365, months366

    months365 = np.concatenate([np.arange(1, val + 1) for val in MONTHS365])
    months366 = np.concatenate([np.arange(1, val + 1) for val in MONTHS366])

    for i in range(800):
        if i % 4 != 0:
            for j in range(365):
                days_to_day[idx + j] = months365[j]
            idx += 365
        elif i % 100 != 0:
            for j in range(366):
                days_to_day[idx + j] = months366[j]
            idx += 366
        elif i % 400 == 0:
            for j in range(366):
                days_to_day[idx + j] = months366[j]
            idx += 366
        else:
            for j in range(365):
                days_to_day[idx + j] = months365[j]
            idx += 365
    return days_to_day

def create_days_to_year_day():
    cdef:
        Py_ssize_t i, j
        long idx = 0
        ndarray[np.int64_t] days_to_day = np.empty(366 * 800, dtype='int64')

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
    cdef:
        Py_ssize_t i, j
        long idx = 0
        ndarray[np.int64_t] days_in_month = np.empty(366 * 800, dtype='int64')
        ndarray[np.int64_t] months365, months366

    months365 = np.repeat(np.arange(12), MONTHS365)
    months366 = np.repeat(np.arange(12), MONTHS366)

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
    cdef:
        Py_ssize_t i, j
        long idx = 0
        ndarray[np.uint8_t, cast=True] days_to_leap = np.zeros(366 * 800, dtype='bool')

    for i in range(800):
        if (i % 4 == 0) and (i % 100 != 0 or i % 400 == 0):
            for j in range(366):
                days_to_leap[idx + j] = True
            idx += 366
        else:
            idx += 365

    return days_to_leap

def create_days_to_quarter_start():
    cdef:
        Py_ssize_t i, j
        long idx = 0
        ndarray[np.uint8_t, cast=True] days_to_day = np.zeros(366 * 800, dtype='bool')

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
    cdef:
        Py_ssize_t i, j
        long idx = 0
        ndarray[np.uint8_t, cast=True] days_to_day = np.zeros(366 * 800, dtype='bool')

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
    cdef:
        Py_ssize_t i, j
        long idx = 0
        ndarray[np.uint8_t, cast=True] days_to_day = np.zeros(366 * 800, dtype='bool')

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
    cdef:
        Py_ssize_t i, j
        long idx = 0
        ndarray[np.uint8_t, cast=True] days_to_day = np.zeros(366 * 800, dtype='bool')

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

def get_weeks_in_year():
    cdef:
        Py_ssize_t i
        int n
        ndarray[np.uint8_t, cast=True] result = np.zeros(800, dtype='bool')
        ndarray[np.int64_t] a = np.array([4, 9, 15, 20, 26, 32, 37, 43, 48, 54, 60, 65,
                                          71, 76, 82, 88, 93, 99, 105, 111, 116, 122,
                                          128, 133, 139, 144, 150, 156, 161, 167, 172,
                                          178, 184, 189, 195, 201, 207, 212, 218, 224,
                                          229, 235, 240, 246, 252, 257, 263, 268, 274,
                                          280, 285, 291, 296, 303, 308, 314, 320, 325,
                                          331, 336, 342, 348, 353, 359, 364, 370, 376,
                                          381, 387, 392, 398], dtype='int64')
    n = len(a)
    for i in range(n):
        result[a[i]] = True
        result[400 + a[i]] = True
    return result

@cython.cdivision(True)
def create_days_to_week_num():
    cdef:
        Py_ssize_t i, j
        int idx = 366, wiy, wiy_prev = 52, wn, dow
        ndarray[np.int64_t] days_to_week_num = np.empty(366 * 700, dtype='int64')
        ndarray[np.uint8_t, cast=True] weeks_in_year = get_weeks_in_year()

    for i in range(1, 700):
        wiy = weeks_in_year[i] + 52
        if i % 4 != 0:
            for j in range(365):
                dow = (idx + j + 5) % 7 + 1
                wn = (j + 1 - dow + 10) // 7
                if wn == 0:
                    days_to_week_num[idx + j] = wiy_prev
                elif wn > wiy:
                    days_to_week_num[idx + j] = 1
                else:
                    days_to_week_num[idx + j] = wn
            idx += 365
        elif i % 100 != 0:
            for j in range(366):
                dow = (idx + j + 5) % 7 + 1
                wn = (j + 1 - dow + 10) // 7
                if wn == 0:
                    days_to_week_num[idx + j] = wiy_prev
                elif wn > wiy:
                    days_to_week_num[idx + j] = 1
                else:
                    days_to_week_num[idx + j] = wn

            idx += 366
        elif i % 400 == 0:
            for j in range(366):
                dow = (idx + j + 5) % 7 + 1
                wn = (j + 1 - dow + 10) // 7
                if wn == 0:
                    days_to_week_num[idx + j] = wiy_prev
                elif wn > wiy:
                    days_to_week_num[idx + j] = 1
                else:
                    days_to_week_num[idx + j] = wn
            idx += 366
        else:
            for j in range(365):
                dow = (idx + j + 5) % 7 + 1
                wn = (j + 1 - dow + 10) // 7
                if wn == 0:
                    days_to_week_num[idx + j] = wiy_prev
                elif wn > wiy:
                    days_to_week_num[idx + j] = 1
                else:
                    days_to_week_num[idx + j] = wn
            idx += 365
        wiy_prev = wiy

    # days_to_week_num[idx:idx + idx - 365] = days_to_week_num[365:idx]

    return days_to_week_num

def create_days_to_nano_years():
    cdef:
        Py_ssize_t i, j
        ndarray[np.int64_t] days_to_nanos = np.zeros(366 * 662, dtype='int64')
        np.int64_t cur_nanos = 0
        int DAYS_SINCE_1600 = 135140, idx = DAYS_SINCE_1600
        long n365 = 365 * 86400 * 10 ** 9, n366 = 366 * 86400 * 10 ** 9

    for i in range(1970, 2263):
        if i % 4 != 0:
            for j in range(365):
                days_to_nanos[idx + j] = cur_nanos
            idx += 365
            cur_nanos += n365

        elif i % 100 != 0 or i % 400 == 0:
            for j in range(366):
                days_to_nanos[idx + j] = cur_nanos
            idx += 366
            cur_nanos += n366
        else:
            for j in range(365):
                days_to_nanos[idx + j] = cur_nanos
            idx += 365
            cur_nanos += n365

    idx = DAYS_SINCE_1600
    cur_nanos = 0
    for i in range(1969, 1679, -1):
        if i % 4 != 0:
            cur_nanos -= n365
            idx -= 365
            for j in range(365):
                days_to_nanos[idx + j] = cur_nanos
        elif i % 100 != 0 or i % 400 == 0:
            cur_nanos -= n366
            idx -= 366
            for j in range(366):
                days_to_nanos[idx + j] = cur_nanos
        else:
            cur_nanos -= n365
            idx -= 365
            for j in range(365):
                days_to_nanos[idx + j] = cur_nanos
    return days_to_nanos

@cython.cdivision(True)
def weekday_name(ndarray[np.int64_t, ndim=2] a):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0], nc = a.shape[1]
        np.int64_t NAT = np.datetime64('nat').astype('int64')
        ndarray[object, ndim=2] result = np.empty((nr, nc), dtype='O')
        ndarray[object] day_count = np.array(['Thursday', 'Friday', 'Saturday', 'Sunday',
                                               'Monday', 'Tuesday', 'Wednesday'], 'O')

    for j in range(nc):
        for i in range(nr):
            if a[i, j] == NAT:
                result[i, j] = None
            else:
                result[i, j] = day_count[a[i, j] / YEAR_NANOS % 7]
    return result

def day_of_week(ndarray[np.int64_t, ndim=2] a):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0], nc = a.shape[1]
        ndarray[np.float64_t, ndim=2] result = np.empty((nr, nc), dtype='float64')
        np.int64_t NAT = np.datetime64('nat').astype('int64')

    for j in range(nc):
        for i in range(nr):
            if a[i, j] == NAT:
                result[i, j] = nan
            else:
                result[i, j] = (a[i, j] / YEAR_NANOS + 3) % 7
    return result

def month(ndarray[np.int64_t, ndim=2] a):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0], nc = a.shape[1]
        ndarray[np.float64_t, ndim=2] result = np.empty((nr, nc), dtype='float64')
        np.int64_t NAT = np.datetime64('nat').astype('int64')

    for j in range(nc):
        for i in range(nr):
            if a[i, j] == NAT:
                result[i, j] = nan
            else:
                result[i, j] = a[i, j] % 12 + 1
    return result

def month2(ndarray[np.int64_t, ndim=2] a):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0], nc = a.shape[1]
        ndarray[np.float64_t, ndim=2] result = np.empty((nr, nc), dtype='float64')
        np.int64_t NAT = np.datetime64('nat').astype('int64')
        ndarray[np.int64_t] days_to_month
        int DAYS_SINCE_1600 = 135140

    days_to_month = create_days_to_month()

    for j in range(nc):
        for i in range(nr):
            if a[i, j] == NAT:
                result[i, j] = nan
            else:
                result[i, j] = days_to_month[<long> (a[i, j] / YEAR_NANOS) + DAYS_SINCE_1600]
    return result

def quarter(ndarray[np.int64_t, ndim=2] a):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0], nc = a.shape[1]
        ndarray[np.float64_t, ndim=2] result = np.empty((nr, nc), dtype='float64')
        np.int64_t NAT = np.datetime64('nat').astype('int64')
        ndarray[np.int64_t] days_to_month
        int DAYS_SINCE_1600 = 135140

    days_to_month = create_days_to_month()

    for j in range(nc):
        for i in range(nr):
            if a[i, j] == NAT:
                result[i, j] = nan
            else:
                result[i, j] = days_to_month[<long> (a[i, j] / YEAR_NANOS) + DAYS_SINCE_1600] % 12 / 3 + 1
    return result

def nanosecond(ndarray[np.int64_t, ndim=2] a):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0], nc = a.shape[1]
        ndarray[np.float64_t, ndim=2] result = np.empty((nr, nc), dtype='float64')
        np.int64_t NAT = np.datetime64('nat').astype('int64')

    for j in range(nc):
        for i in range(nr):
            if a[i, j] == NAT:
                result[i, j] = nan
            else:
                result[i, j] = a[i, j] % 1000
    return result

def microsecond(ndarray[np.int64_t, ndim=2] a):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0], nc = a.shape[1]
        ndarray[np.float64_t, ndim=2] result = np.empty((nr, nc), dtype='float64')

    for j in range(nc):
        for i in range(nr):
            if a[i, j] == NAT:
                result[i, j] = nan
            else:
                result[i, j] = <long> (a[i, j] / 1000) % 1000
    return result


def millisecond(ndarray[np.int64_t, ndim=2] a):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0], nc = a.shape[1]
        ndarray[np.float64_t, ndim=2] result = np.empty((nr, nc), dtype='float64')

    for j in range(nc):
        for i in range(nr):
            if a[i, j] == NAT:
                result[i, j] = nan
            else:
                result[i, j] = <long> (a[i, j] / 1000000) % 1000
    return result

def second(ndarray[np.int64_t, ndim=2] a):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0], nc = a.shape[1]
        ndarray[np.float64_t, ndim=2] result = np.empty((nr, nc), dtype='float64')

    for j in range(nc):
        for i in range(nr):
            if a[i, j] == NAT:
                result[i, j] = nan
            else:
                result[i, j] = <long> (a[i, j] / 10 ** 9) % 60
    return result

def minute(ndarray[np.int64_t, ndim=2] a):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0], nc = a.shape[1]
        ndarray[np.float64_t, ndim=2] result = np.empty((nr, nc), dtype='float64')
        long nanos = 10 ** 9 * 60

    for j in range(nc):
        for i in range(nr):
            if a[i, j] == NAT:
                result[i, j] = nan
            else:
                result[i, j] = <long> (a[i, j] / nanos) % 60
    return result

def hour(ndarray[np.int64_t, ndim=2] a):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0], nc = a.shape[1]
        ndarray[np.float64_t, ndim=2] result = np.empty((nr, nc), dtype='float64')
        long nanos = 10 ** 9 * 60

    for j in range(nc):
        for i in range(nr):
            if a[i, j] == NAT:
                result[i, j] = nan
            else:
                result[i, j] = <long> (a[i, j] / nanos) % 24
    return result

def year(ndarray[np.int64_t, ndim=2] a):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0], nc = a.shape[1]
        ndarray[np.float64_t, ndim=2] result = np.empty((nr, nc), dtype='float64')
        ndarray[np.int64_t] days_to_year

    days_to_year = create_days_to_year()

    for j in range(nc):
        for i in range(nr):
            if a[i, j] == NAT:
                result[i, j] = nan
            else:
                result[i, j] = days_to_year[<long> (a[i, j] / YEAR_NANOS) + DAYS_SINCE_1600]
    return result

def day(ndarray[np.int64_t, ndim=2] a):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0], nc = a.shape[1]
        ndarray[np.float64_t, ndim=2] result = np.empty((nr, nc), dtype='float64')
        ndarray[np.int64_t] days_to_year

    days_to_year = create_days_to_day()

    for j in range(nc):
        for i in range(nr):
            if a[i, j] == NAT:
                result[i, j] = nan
            else:
                result[i, j] = days_to_year[<long> (a[i, j] / YEAR_NANOS) + DAYS_SINCE_1600]
    return result

def day_of_year(ndarray[np.int64_t, ndim=2] a):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0], nc = a.shape[1]
        ndarray[np.float64_t, ndim=2] result = np.empty((nr, nc), dtype='float64')
        ndarray[np.int64_t] days_to_year_day

    days_to_year_day = create_days_to_year_day()

    for j in range(nc):
        for i in range(nr):
            if a[i, j] == NAT:
                result[i, j] = nan
            else:
                result[i, j] = days_to_year_day[<long> (a[i, j] / YEAR_NANOS) + DAYS_SINCE_1600]
    return result


def days_in_month(ndarray[np.int64_t, ndim=2] a):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0], nc = a.shape[1]
        ndarray[np.float64_t, ndim=2] result = np.empty((nr, nc), dtype='float64')
        ndarray[np.int64_t] days_to_days_in_month

    days_to_days_in_month = create_days_in_month()

    for j in range(nc):
        for i in range(nr):
            if a[i, j] == NAT:
                result[i, j] = nan
            else:
                result[i, j] = days_to_days_in_month[<long> (a[i, j] / YEAR_NANOS) + DAYS_SINCE_1600]
    return result

def is_leap_year(ndarray[np.int64_t, ndim=2] a):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0], nc = a.shape[1]
        ndarray[np.uint8_t, cast=True, ndim=2] result = np.empty((nr, nc), dtype='bool')
        ndarray[np.uint8_t, cast=True] days_to_is_leap_year

    days_to_is_leap_year = create_days_to_leap_year()

    for j in range(nc):
        for i in range(nr):
            if a[i, j] == NAT:
                result[i, j] = False
            else:
                result[i, j] = days_to_is_leap_year[<long> (a[i, j] / YEAR_NANOS) + DAYS_SINCE_1600]
    return result

def is_quarter_start(ndarray[np.int64_t, ndim=2] a):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0], nc = a.shape[1]
        ndarray[np.uint8_t, cast=True, ndim=2] result = np.empty((nr, nc), dtype='bool')
        ndarray[np.uint8_t, cast=True] days_to_is_quarter_start

    days_to_is_quarter_start = create_days_to_quarter_start()

    for j in range(nc):
        for i in range(nr):
            if a[i, j] == NAT:
                result[i, j] = False
            else:
                result[i, j] = days_to_is_quarter_start[<long> (a[i, j] / YEAR_NANOS) + DAYS_SINCE_1600]
    return result

def is_quarter_end(ndarray[np.int64_t, ndim=2] a):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0], nc = a.shape[1]
        ndarray[np.uint8_t, cast=True, ndim=2] result = np.empty((nr, nc), dtype='bool')
        ndarray[np.uint8_t, cast=True] days_to_is_quarter_end

    days_to_is_quarter_end = create_days_to_quarter_end()

    for j in range(nc):
        for i in range(nr):
            if a[i, j] == NAT:
                result[i, j] = False
            else:
                result[i, j] = days_to_is_quarter_end[<long> (a[i, j] / YEAR_NANOS) + DAYS_SINCE_1600]
    return result


def is_year_end(ndarray[np.int64_t, ndim=2] a):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0], nc = a.shape[1]
        ndarray[np.uint8_t, cast=True, ndim=2] result = np.empty((nr, nc), dtype='bool')
        ndarray[np.uint8_t, cast=True] days_to_is_year_end

    days_to_is_year_end = create_days_to_year_end()

    for j in range(nc):
        for i in range(nr):
            if a[i, j] == NAT:
                result[i, j] = False
            else:
                result[i, j] = days_to_is_year_end[<long> (a[i, j] / YEAR_NANOS) + DAYS_SINCE_1600]
    return result

def is_year_start(ndarray[np.int64_t, ndim=2] a):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0], nc = a.shape[1]
        ndarray[np.uint8_t, cast=True, ndim=2] result = np.empty((nr, nc), dtype='bool')
        ndarray[np.uint8_t, cast=True] days_to_is_year_start

    days_to_is_year_start = create_days_to_year_start()

    for j in range(nc):
        for i in range(nr):
            if a[i, j] == NAT:
                result[i, j] = False
            else:
                result[i, j] = days_to_is_year_start[<long> (a[i, j] / YEAR_NANOS) + DAYS_SINCE_1600]
    return result

@cython.cdivision(True)
def week_of_year(ndarray[np.float64_t, ndim=2] week, ndarray[np.float64_t, ndim=2] year):
    cdef:
        Py_ssize_t i, j
        int nr = week.shape[0], nc = week.shape[1]
        ndarray[np.float64_t, ndim=2] result = np.empty((nr, nc), dtype='float64')
        ndarray[np.uint8_t, cast=True] has_53_weeks

    has_53_weeks = get_weeks_in_year()

    for j in range(nc):
        for i in range(nr):
            if week[i, j] == 0:
                result[i, j] = has_53_weeks[<int> ((year[i, j] - 1) % 400)] + 52
            elif week[i, j] == 53:
                if has_53_weeks[<int> (year[i, j] % 400)]:
                    result[i, j] = 53
                else:
                    result[i, j] = 1
            elif week[i, j] > 0:
                result[i, j] = week[i, j]
            else:
                result[i, j] = nan
    return result

@cython.cdivision(True)
def week_of_year2(ndarray[np.int64_t, ndim=2] a):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0], nc = a.shape[1]
        ndarray[np.float64_t, ndim=2] result = np.empty((nr, nc), dtype='float64')
        ndarray[np.int64_t] weeks_in_year

    weeks_in_year = create_days_to_week_num()

    for j in range(nc):
        for i in range(nr):
            if a[i, j] == NAT:
                result[i, j] = nan
            else:
                result[i, j] = weeks_in_year[a[i, j] // YEAR_NANOS + DAYS_SINCE_1600]
    return result

def floor_us(ndarray[np.int64_t, ndim=2] a):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0], nc = a.shape[1]
        ndarray[np.int64_t, ndim=2] result = np.empty((nr, nc), dtype='int64')

    for j in range(nc):
        for i in range(nr):
            result[i, j] = a[i, j] // 1000 * 1000
    return result.astype('datetime64[ns]')

def floor_ms(ndarray[np.int64_t, ndim=2] a):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0], nc = a.shape[1]
        ndarray[np.int64_t, ndim=2] result = np.empty((nr, nc), dtype='int64')

    for j in range(nc):
        for i in range(nr):
            result[i, j] = a[i, j] // 1000000 * 1000000
    return result.astype('datetime64[ns]')

def floor_s(ndarray[np.int64_t, ndim=2] a):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0], nc = a.shape[1]
        ndarray[np.int64_t, ndim=2] result = np.empty((nr, nc), dtype='int64')

    for j in range(nc):
        for i in range(nr):
            if a[i, j] == NAT:
                result[i, j] = NAT
            else:
                result[i, j] = a[i, j] // 10 ** 9 * 10 ** 9
    return result.astype('datetime64[ns]')

def floor_m(ndarray[np.int64_t, ndim=2] a):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0], nc = a.shape[1]
        ndarray[np.int64_t, ndim=2] result = np.empty((nr, nc), dtype='int64')
        long nanos = 10 ** 9 * 60

    for j in range(nc):
        for i in range(nr):
            if a[i, j] == NAT:
                result[i, j] = NAT
            else:
                result[i, j] = a[i, j] // nanos * nanos
    return result.astype('datetime64[ns]')


def floor_h(ndarray[np.int64_t, ndim=2] a):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0], nc = a.shape[1]
        ndarray[np.int64_t, ndim=2] result = np.empty((nr, nc), dtype='int64')
        long nanos = 10 ** 9 * 3600

    for j in range(nc):
        for i in range(nr):
            if a[i, j] == NAT:
                result[i, j] = NAT
            else:
                result[i, j] = a[i, j] // nanos * nanos
    return result.astype('datetime64[ns]')

def floor_D(ndarray[np.int64_t, ndim=2] a):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0], nc = a.shape[1]
        ndarray[np.int64_t, ndim=2] result = np.empty((nr, nc), dtype='int64')
        long nanos = 10 ** 9 * 3600 * 24

    for j in range(nc):
        for i in range(nr):
            if a[i, j] == NAT:
                result[i, j] = NAT
            else:
                result[i, j] = a[i, j] // nanos * nanos
    return result.astype('datetime64[ns]')

def floor_Y(ndarray[np.int64_t, ndim=2] a):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0], nc = a.shape[1]
        ndarray[np.int64_t, ndim=2] result = np.empty((nr, nc), dtype='int64')
        ndarray[np.int64_t] days_to_nano

    if a.size < 5000:
        return a.astype('datetime64[Y]').astype('datetime64[ns]')

    days_to_nano = create_days_to_nano_years()

    for j in range(nc):
        for i in range(nr):
            if a[i, j] == NAT:
                result[i, j] = NAT
            else:
                result[i, j] = days_to_nano[a[i, j] // YEAR_NANOS + DAYS_SINCE_1600]
    return result.astype('datetime64[ns]')

def ceil_us(ndarray[np.float64_t, ndim=2] a):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0], nc = a.shape[1]
        ndarray[np.int64_t, ndim=2] result = np.empty((nr, nc), dtype='int64')
        np.int64_t nanos = 10 ** 3

    for j in range(nc):
        for i in range(nr):
            if a[i, j] == NAT:
                result[i, j] = NAT
            else:
                result[i, j] = <long> (ceil(a[i, j] / nanos)) * nanos
    return result.astype('datetime64[ns]')

def ceil_ms(ndarray[np.float64_t, ndim=2] a):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0], nc = a.shape[1]
        ndarray[np.float64_t, ndim=2] result = np.empty((nr, nc), dtype='float64')
        np.int64_t nanos = 10 ** 6

    for j in range(nc):
        for i in range(nr):
            if a[i, j] == NAT:
                result[i, j] = NAT
            else:
                result[i, j] = <long> (ceil(a[i, j] / nanos)) * nanos
    return result.astype('datetime64[ns]')

def ceil_s(ndarray[np.float64_t, ndim=2] a):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0], nc = a.shape[1]
        ndarray[np.float64_t, ndim=2] result = np.empty((nr, nc), dtype='float64')
        np.int64_t nanos = 10 ** 9

    for j in range(nc):
        for i in range(nr):
            if a[i, j] == NAT:
                result[i, j] = NAT
            else:
                result[i, j] = <long> (ceil(a[i, j] / nanos)) * nanos
    return result.astype('datetime64[ns]')

def ceil_m(ndarray[np.float64_t, ndim=2] a):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0], nc = a.shape[1]
        ndarray[np.float64_t, ndim=2] result = np.empty((nr, nc), dtype='float64')
        np.int64_t nanos = 10 ** 9 * 60

    for j in range(nc):
        for i in range(nr):
            if a[i, j] == NAT:
                result[i, j] = NAT
            else:
                result[i, j] = <long> (ceil(a[i, j] / nanos)) * nanos
    return result.astype('datetime64[ns]')

def ceil_h(ndarray[np.float64_t, ndim=2] a):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0], nc = a.shape[1]
        ndarray[np.float64_t, ndim=2] result = np.empty((nr, nc), dtype='float64')
        np.int64_t nanos = 10 ** 9 * 3600

    for j in range(nc):
        for i in range(nr):
            if a[i, j] == NAT:
                result[i, j] = NAT
            else:
                result[i, j] = <long> (ceil(a[i, j] / nanos)) * nanos
    return result.astype('datetime64[ns]')

def ceil_D(ndarray[np.float64_t, ndim=2] a):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0], nc = a.shape[1]
        ndarray[np.float64_t, ndim=2] result = np.empty((nr, nc), dtype='float64')
        np.int64_t nanos = 10 ** 9 * 86400

    for j in range(nc):
        for i in range(nr):
            if a[i, j] == NAT:
                result[i, j] = NAT
            else:
                result[i, j] = <long> (ceil(a[i, j] / nanos)) * nanos
    return result.astype('datetime64[ns]')

def ceil_Y(ndarray[np.float64_t, ndim=2] a):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0], nc = a.shape[1]
        ndarray[np.float64_t, ndim=2] result = np.empty((nr, nc), dtype='float64')
        ndarray[np.int64_t] days_to_nano
        long cur_year_nanos, idx

    days_to_nano = create_days_to_nano_years()

    for j in range(nc):
        for i in range(nr):
            if a[i, j] == NAT:
                result[i, j] = NAT
            else:
                idx = <long> (a[i, j] // YEAR_NANOS) + DAYS_SINCE_1600
                cur_year_nanos = days_to_nano[idx]
                if cur_year_nanos == a[i, j]:
                    result[i, j] = a[i, j]
                else:
                    idx += 365
                    while days_to_nano[idx] == cur_year_nanos:
                        idx += 1
                    result[i, j] = days_to_nano[idx]
    return result.astype('datetime64[ns]')

def to_pytime(ndarray[np.float64_t, ndim=2] a):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0], nc = a.shape[1]
        ndarray[object, ndim=2] result = np.empty((nr, nc), dtype='O')
        long day_nanos = 86400

    for j in range(nc):
        for i in range(nr):
            if a[i, j] == NAT:
                result[i, j] = None
            else:
                result[i, j] = datetime.datetime.fromtimestamp(a[i, j] / 10 ** 9 + 18000).time()
    return result


def strftime(ndarray[np.float64_t, ndim=2] a, str date_format):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0], nc = a.shape[1]
        ndarray[object, ndim=2] result = np.empty((nr, nc), dtype='O')
        long day_nanos = 86400

    for j in range(nc):
        for i in range(nr):
            if a[i, j] == NAT:
                result[i, j] = None
            else:
                result[i, j] = datetime.datetime.fromtimestamp(a[i, j] / 10 ** 9 + 18000).strftime(date_format)
    return result

def round_us(ndarray[np.float64_t, ndim=2] a):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0], nc = a.shape[1]
        ndarray[np.int64_t, ndim=2] result = np.empty((nr, nc), dtype='int64')

    for j in range(nc):
        for i in range(nr):
            result[i, j] = <long> round(a[i, j] / 1000) * 1000
    return result.astype('datetime64[ns]')

def round_ms(ndarray[np.float64_t, ndim=2] a):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0], nc = a.shape[1]
        ndarray[np.int64_t, ndim=2] result = np.empty((nr, nc), dtype='int64')

    for j in range(nc):
        for i in range(nr):
            result[i, j] = <long> round(a[i, j] / 1000000) * 1000000
    return result.astype('datetime64[ns]')

@cython.cdivision(True)
def round_s(ndarray[np.float64_t, ndim=2] a):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0], nc = a.shape[1]
        ndarray[np.int64_t, ndim=2] result = np.empty((nr, nc), dtype='int64')

    for j in range(nc):
        for i in range(nr):
            if a[i, j] == NAT:
                result[i, j] = NAT
            else:
                result[i, j] = <long> round(a[i, j] / 10 ** 9) * 10 ** 9
    return result.astype('datetime64[ns]')

def round_m(ndarray[np.float64_t, ndim=2] a):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0], nc = a.shape[1]
        ndarray[np.int64_t, ndim=2] result = np.empty((nr, nc), dtype='int64')
        long nanos = 10 ** 9 * 60

    for j in range(nc):
        for i in range(nr):
            if a[i, j] == NAT:
                result[i, j] = NAT
            else:
                result[i, j] = <long> round(a[i, j] / nanos) * nanos
    return result.astype('datetime64[ns]')


def round_h(ndarray[np.float64_t, ndim=2] a):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0], nc = a.shape[1]
        ndarray[np.int64_t, ndim=2] result = np.empty((nr, nc), dtype='int64')
        long nanos = 10 ** 9 * 3600

    for j in range(nc):
        for i in range(nr):
            if a[i, j] == NAT:
                result[i, j] = NAT
            else:
                result[i, j] = <long> round(a[i, j] / nanos) * nanos
    return result.astype('datetime64[ns]')

def round_D(ndarray[np.float64_t, ndim=2] a):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0], nc = a.shape[1]
        ndarray[np.int64_t, ndim=2] result = np.empty((nr, nc), dtype='int64')
        long nanos = 10 ** 9 * 3600 * 24

    for j in range(nc):
        for i in range(nr):
            if a[i, j] == NAT:
                result[i, j] = NAT
            else:
                result[i, j] = <long> round(a[i, j] / nanos) * nanos
    return result.astype('datetime64[ns]')

def round_Y(ndarray[np.float64_t, ndim=2] a):
    cdef:
        Py_ssize_t i, j
        int nr = a.shape[0], nc = a.shape[1]
        ndarray[np.float64_t, ndim=2] result = np.empty((nr, nc), dtype='float64')
        ndarray[np.int64_t] days_to_nano
        long cur_year_nanos, idx

    days_to_nano = create_days_to_nano_years()

    for j in range(nc):
        for i in range(nr):
            if a[i, j] == NAT:
                result[i, j] = NAT
            else:
                idx = <long> (a[i, j] // YEAR_NANOS) + DAYS_SINCE_1600
                cur_year_nanos = days_to_nano[idx]
                idx += 365
                while days_to_nano[idx] == cur_year_nanos:
                    idx += 1
                if a[i, j] - cur_year_nanos > days_to_nano[idx] - a[i, j]:
                    result[i, j] = days_to_nano[idx]
                else:
                    result[i, j] = cur_year_nanos
    return result.astype('datetime64[ns]')