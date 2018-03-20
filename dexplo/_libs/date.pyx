import numpy as np
from numpy import nan
from libc.math cimport ceil, round
from numpy cimport ndarray
cimport numpy as np
import cython
import datetime


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

def get_weeks_in_year():
    cdef int i, n
    cdef ndarray[np.uint8_t, cast=True] result = np.zeros(800, dtype='bool')
    cdef ndarray[np.int64_t] a = np.array([4, 9, 15, 20, 26, 32, 37, 43, 48, 54, 60, 65,
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
    cdef long i, j, idx = 366
    cdef ndarray[np.int64_t] days_to_week_num = np.empty(366 * 700, dtype='int64')
    cdef ndarray[np.uint8_t, cast=True] weeks_in_year = get_weeks_in_year()
    cdef int wiy, wiy_prev = 52, wn, dow

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
    cdef long i, j
    cdef ndarray[np.int64_t] days_to_nanos = np.zeros(366 * 662, dtype='int64')
    cdef long n365 = 365 * 86400 * 10 ** 9
    cdef long n366 = 366 * 86400 * 10 ** 9
    cdef np.int64_t cur_nanos = 0
    cdef int days_since_1600 = 135140
    cdef int idx = days_since_1600

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

    idx = days_since_1600
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
    cdef int i, j
    cdef int nr = a.shape[0]
    cdef int nc = a.shape[1]
    cdef np.int64_t NAT = np.datetime64('nat').astype('int64')
    cdef long year_nanos = 10 ** 9 * 86400
    cdef ndarray[object, ndim=2] result = np.empty((nr, nc), dtype='O')
    cdef ndarray[object] day_count = np.array(['Thursday', 'Friday', 'Saturday', 'Sunday',
                                               'Monday', 'Tuesday', 'Wednesday'], 'O')

    for j in range(nc):
        for i in range(nr):
            if a[i, j] == NAT:
                result[i, j] = None
            else:
                result[i, j] = day_count[a[i, j] / year_nanos % 7]
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

def quarter(ndarray[np.int64_t, ndim=2] a):
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
                result[i, j] = days_to_month[<long> (a[i, j] / year_nanos) + days_since_1600] % 12 / 3 + 1
    return result

def nanosecond(ndarray[np.int64_t, ndim=2] a):
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
                result[i, j] = a[i, j] % 1000
    return result

def microsecond(ndarray[np.int64_t, ndim=2] a):
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
                result[i, j] = <long> (a[i, j] / 1000) % 1000
    return result


def millisecond(ndarray[np.int64_t, ndim=2] a):
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
                result[i, j] = <long> (a[i, j] / 1000000) % 1000
    return result

def second(ndarray[np.int64_t, ndim=2] a):
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
                result[i, j] = <long> (a[i, j] / 10 ** 9) % 60
    return result

def minute(ndarray[np.int64_t, ndim=2] a):
    cdef int i, j
    cdef int nr = a.shape[0]
    cdef int nc = a.shape[1]
    cdef ndarray[np.float64_t, ndim=2] result = np.empty((nr, nc), dtype='float64')
    cdef np.int64_t NAT = np.datetime64('nat').astype('int64')
    cdef long nanos = 10 ** 9 * 60

    for j in range(nc):
        for i in range(nr):
            if a[i, j] == NAT:
                result[i, j] = nan
            else:
                result[i, j] = <long> (a[i, j] / nanos) % 60
    return result

def hour(ndarray[np.int64_t, ndim=2] a):
    cdef int i, j
    cdef int nr = a.shape[0]
    cdef int nc = a.shape[1]
    cdef ndarray[np.float64_t, ndim=2] result = np.empty((nr, nc), dtype='float64')
    cdef np.int64_t NAT = np.datetime64('nat').astype('int64')
    cdef long nanos = 10 ** 9 * 3600

    for j in range(nc):
        for i in range(nr):
            if a[i, j] == NAT:
                result[i, j] = nan
            else:
                result[i, j] = <long> (a[i, j] / nanos) % 24
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

@cython.cdivision(True)
def week_of_year(ndarray[np.float64_t, ndim=2] week, ndarray[np.float64_t, ndim=2] year):
    cdef int i, j
    cdef int nr = week.shape[0]
    cdef int nc = week.shape[1]
    cdef ndarray[np.float64_t, ndim=2] result = np.empty((nr, nc), dtype='float64')
    cdef ndarray[np.uint8_t, cast=True] has_53_weeks

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
    cdef int i, j
    cdef int nr = a.shape[0]
    cdef int nc = a.shape[1]
    cdef ndarray[np.float64_t, ndim=2] result = np.empty((nr, nc), dtype='float64')
    cdef np.int64_t NAT = np.datetime64('nat').astype('int64')
    cdef ndarray[np.int64_t] weeks_in_year
    cdef int days_since_1600 = 135140
    cdef long year_nanos = 10 ** 9 * 86400

    weeks_in_year = create_days_to_week_num()

    for j in range(nc):
        for i in range(nr):
            if a[i, j] == NAT:
                result[i, j] = nan
            else:
                result[i, j] = weeks_in_year[a[i, j] // year_nanos + days_since_1600]
    return result

def floor_us(ndarray[np.int64_t, ndim=2] a):
    cdef int i, j
    cdef int nr = a.shape[0]
    cdef int nc = a.shape[1]
    cdef ndarray[np.int64_t, ndim=2] result = np.empty((nr, nc), dtype='int64')
    cdef np.int64_t NAT = np.datetime64('nat').astype('int64')

    for j in range(nc):
        for i in range(nr):
            result[i, j] = a[i, j] // 1000 * 1000
    return result.astype('datetime64[ns]')

def floor_ms(ndarray[np.int64_t, ndim=2] a):
    cdef int i, j
    cdef int nr = a.shape[0]
    cdef int nc = a.shape[1]
    cdef ndarray[np.int64_t, ndim=2] result = np.empty((nr, nc), dtype='int64')
    cdef np.int64_t NAT = np.datetime64('nat').astype('int64')

    for j in range(nc):
        for i in range(nr):
            result[i, j] = a[i, j] // 1000000 * 1000000
    return result.astype('datetime64[ns]')

def floor_s(ndarray[np.int64_t, ndim=2] a):
    cdef int i, j
    cdef int nr = a.shape[0]
    cdef int nc = a.shape[1]
    cdef ndarray[np.int64_t, ndim=2] result = np.empty((nr, nc), dtype='int64')
    cdef np.int64_t NAT = np.datetime64('nat').astype('int64')

    for j in range(nc):
        for i in range(nr):
            if a[i, j] == NAT:
                result[i, j] = NAT
            else:
                result[i, j] = a[i, j] // 10 ** 9 * 10 ** 9
    return result.astype('datetime64[ns]')

def floor_m(ndarray[np.int64_t, ndim=2] a):
    cdef int i, j
    cdef int nr = a.shape[0]
    cdef int nc = a.shape[1]
    cdef ndarray[np.int64_t, ndim=2] result = np.empty((nr, nc), dtype='int64')
    cdef np.int64_t NAT = np.datetime64('nat').astype('int64')
    cdef long nanos = 10 ** 9 * 60

    for j in range(nc):
        for i in range(nr):
            if a[i, j] == NAT:
                result[i, j] = NAT
            else:
                result[i, j] = a[i, j] // nanos * nanos
    return result.astype('datetime64[ns]')


def floor_h(ndarray[np.int64_t, ndim=2] a):
    cdef int i, j
    cdef int nr = a.shape[0]
    cdef int nc = a.shape[1]
    cdef ndarray[np.int64_t, ndim=2] result = np.empty((nr, nc), dtype='int64')
    cdef np.int64_t NAT = np.datetime64('nat').astype('int64')
    cdef long nanos = 10 ** 9 * 3600

    for j in range(nc):
        for i in range(nr):
            if a[i, j] == NAT:
                result[i, j] = NAT
            else:
                result[i, j] = a[i, j] // nanos * nanos
    return result.astype('datetime64[ns]')

def floor_D(ndarray[np.int64_t, ndim=2] a):
    cdef int i, j
    cdef int nr = a.shape[0]
    cdef int nc = a.shape[1]
    cdef ndarray[np.int64_t, ndim=2] result = np.empty((nr, nc), dtype='int64')
    cdef np.int64_t NAT = np.datetime64('nat').astype('int64')
    cdef long nanos = 10 ** 9 * 3600 * 24

    for j in range(nc):
        for i in range(nr):
            if a[i, j] == NAT:
                result[i, j] = NAT
            else:
                result[i, j] = a[i, j] // nanos * nanos
    return result.astype('datetime64[ns]')

def floor_Y(ndarray[np.int64_t, ndim=2] a):
    cdef int i, j
    cdef int nr = a.shape[0]
    cdef int nc = a.shape[1]
    cdef ndarray[np.int64_t, ndim=2] result = np.empty((nr, nc), dtype='int64')
    cdef np.int64_t NAT = np.datetime64('nat').astype('int64')
    cdef ndarray[np.int64_t] days_to_nano
    cdef int days_since_1600 = 135140
    cdef long year_nanos = 10 ** 9 * 86400

    if a.size < 5000:
        return a.astype('datetime64[Y]').astype('datetime64[ns]')

    days_to_nano = create_days_to_nano_years()

    for j in range(nc):
        for i in range(nr):
            if a[i, j] == NAT:
                result[i, j] = NAT
            else:
                result[i, j] = days_to_nano[a[i, j] // year_nanos + days_since_1600]
    return result.astype('datetime64[ns]')

def ceil_us(ndarray[np.float64_t, ndim=2] a):
    cdef int i, j
    cdef int nr = a.shape[0]
    cdef int nc = a.shape[1]
    cdef ndarray[np.int64_t, ndim=2] result = np.empty((nr, nc), dtype='int64')
    cdef np.int64_t NAT = np.datetime64('nat').astype('int64')
    cdef np.int64_t nanos = 10 ** 3

    for j in range(nc):
        for i in range(nr):
            if a[i, j] == NAT:
                result[i, j] = NAT
            else:
                result[i, j] = <long> (ceil(a[i, j] / nanos)) * nanos
    return result.astype('datetime64[ns]')

def ceil_ms(ndarray[np.float64_t, ndim=2] a):
    cdef int i, j
    cdef int nr = a.shape[0]
    cdef int nc = a.shape[1]
    cdef ndarray[np.float64_t, ndim=2] result = np.empty((nr, nc), dtype='float64')
    cdef np.int64_t NAT = np.datetime64('nat').astype('int64')
    cdef np.int64_t nanos = 10 ** 6

    for j in range(nc):
        for i in range(nr):
            if a[i, j] == NAT:
                result[i, j] = NAT
            else:
                result[i, j] = <long> (ceil(a[i, j] / nanos)) * nanos
    return result.astype('datetime64[ns]')

def ceil_s(ndarray[np.float64_t, ndim=2] a):
    cdef int i, j
    cdef int nr = a.shape[0]
    cdef int nc = a.shape[1]
    cdef ndarray[np.float64_t, ndim=2] result = np.empty((nr, nc), dtype='float64')
    cdef np.int64_t NAT = np.datetime64('nat').astype('int64')
    cdef np.int64_t nanos = 10 ** 9

    for j in range(nc):
        for i in range(nr):
            if a[i, j] == NAT:
                result[i, j] = NAT
            else:
                result[i, j] = <long> (ceil(a[i, j] / nanos)) * nanos
    return result.astype('datetime64[ns]')

def ceil_m(ndarray[np.float64_t, ndim=2] a):
    cdef int i, j
    cdef int nr = a.shape[0]
    cdef int nc = a.shape[1]
    cdef ndarray[np.float64_t, ndim=2] result = np.empty((nr, nc), dtype='float64')
    cdef np.int64_t NAT = np.datetime64('nat').astype('int64')
    cdef np.int64_t nanos = 10 ** 9 * 60

    for j in range(nc):
        for i in range(nr):
            if a[i, j] == NAT:
                result[i, j] = NAT
            else:
                result[i, j] = <long> (ceil(a[i, j] / nanos)) * nanos
    return result.astype('datetime64[ns]')

def ceil_h(ndarray[np.float64_t, ndim=2] a):
    cdef int i, j
    cdef int nr = a.shape[0]
    cdef int nc = a.shape[1]
    cdef ndarray[np.float64_t, ndim=2] result = np.empty((nr, nc), dtype='float64')
    cdef np.int64_t NAT = np.datetime64('nat').astype('int64')
    cdef np.int64_t nanos = 10 ** 9 * 3600

    for j in range(nc):
        for i in range(nr):
            if a[i, j] == NAT:
                result[i, j] = NAT
            else:
                result[i, j] = <long> (ceil(a[i, j] / nanos)) * nanos
    return result.astype('datetime64[ns]')

def ceil_D(ndarray[np.float64_t, ndim=2] a):
    cdef int i, j
    cdef int nr = a.shape[0]
    cdef int nc = a.shape[1]
    cdef ndarray[np.float64_t, ndim=2] result = np.empty((nr, nc), dtype='float64')
    cdef np.int64_t NAT = np.datetime64('nat').astype('int64')
    cdef np.int64_t nanos = 10 ** 9 * 86400

    for j in range(nc):
        for i in range(nr):
            if a[i, j] == NAT:
                result[i, j] = NAT
            else:
                result[i, j] = <long> (ceil(a[i, j] / nanos)) * nanos
    return result.astype('datetime64[ns]')

def ceil_Y(ndarray[np.float64_t, ndim=2] a):
    cdef int i, j
    cdef int nr = a.shape[0]
    cdef int nc = a.shape[1]
    cdef ndarray[np.float64_t, ndim=2] result = np.empty((nr, nc), dtype='float64')
    cdef np.int64_t NAT = np.datetime64('nat').astype('int64')
    cdef ndarray[np.int64_t] days_to_nano
    cdef int days_since_1600 = 135140
    cdef long year_nanos = 10 ** 9 * 86400
    cdef long cur_year_nanos, idx

    days_to_nano = create_days_to_nano_years()

    for j in range(nc):
        for i in range(nr):
            if a[i, j] == NAT:
                result[i, j] = NAT
            else:
                idx = <long> (a[i, j] // year_nanos) + days_since_1600
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
    cdef int i, j
    cdef int nr = a.shape[0]
    cdef int nc = a.shape[1]
    cdef ndarray[object, ndim=2] result = np.empty((nr, nc), dtype='O')
    cdef np.int64_t NAT = np.datetime64('nat').astype('int64')
    cdef long day_nanos = 86400


    for j in range(nc):
        for i in range(nr):
            if a[i, j] == NAT:
                result[i, j] = None
            else:
                result[i, j] = datetime.datetime.fromtimestamp(a[i, j] / 10 ** 9 + 18000).time()
    return result


def strftime(ndarray[np.float64_t, ndim=2] a, str date_format):
    cdef int i, j
    cdef int nr = a.shape[0]
    cdef int nc = a.shape[1]
    cdef ndarray[object, ndim=2] result = np.empty((nr, nc), dtype='O')
    cdef np.int64_t NAT = np.datetime64('nat').astype('int64')
    cdef long day_nanos = 86400


    for j in range(nc):
        for i in range(nr):
            if a[i, j] == NAT:
                result[i, j] = None
            else:
                result[i, j] = datetime.datetime.fromtimestamp(a[i, j] / 10 ** 9 + 18000).strftime(date_format)
    return result

def round_us(ndarray[np.float64_t, ndim=2] a):
    cdef int i, j
    cdef int nr = a.shape[0]
    cdef int nc = a.shape[1]
    cdef ndarray[np.int64_t, ndim=2] result = np.empty((nr, nc), dtype='int64')
    cdef np.int64_t NAT = np.datetime64('nat').astype('int64')

    for j in range(nc):
        for i in range(nr):
            result[i, j] = <long> round(a[i, j] / 1000) * 1000
    return result.astype('datetime64[ns]')

def round_ms(ndarray[np.float64_t, ndim=2] a):
    cdef int i, j
    cdef int nr = a.shape[0]
    cdef int nc = a.shape[1]
    cdef ndarray[np.int64_t, ndim=2] result = np.empty((nr, nc), dtype='int64')
    cdef np.int64_t NAT = np.datetime64('nat').astype('int64')

    for j in range(nc):
        for i in range(nr):
            result[i, j] = <long> round(a[i, j] / 1000000) * 1000000
    return result.astype('datetime64[ns]')

@cython.cdivision(True)
def round_s(ndarray[np.float64_t, ndim=2] a):
    cdef int i, j
    cdef int nr = a.shape[0]
    cdef int nc = a.shape[1]
    cdef ndarray[np.int64_t, ndim=2] result = np.empty((nr, nc), dtype='int64')
    cdef np.int64_t NAT = np.datetime64('nat').astype('int64')

    for j in range(nc):
        for i in range(nr):
            if a[i, j] == NAT:
                result[i, j] = NAT
            else:
                result[i, j] = <long> round(a[i, j] / 10 ** 9) * 10 ** 9
    return result.astype('datetime64[ns]')

def round_m(ndarray[np.float64_t, ndim=2] a):
    cdef int i, j
    cdef int nr = a.shape[0]
    cdef int nc = a.shape[1]
    cdef ndarray[np.int64_t, ndim=2] result = np.empty((nr, nc), dtype='int64')
    cdef np.int64_t NAT = np.datetime64('nat').astype('int64')
    cdef long nanos = 10 ** 9 * 60

    for j in range(nc):
        for i in range(nr):
            if a[i, j] == NAT:
                result[i, j] = NAT
            else:
                result[i, j] = <long> round(a[i, j] / nanos) * nanos
    return result.astype('datetime64[ns]')


def round_h(ndarray[np.float64_t, ndim=2] a):
    cdef int i, j
    cdef int nr = a.shape[0]
    cdef int nc = a.shape[1]
    cdef ndarray[np.int64_t, ndim=2] result = np.empty((nr, nc), dtype='int64')
    cdef np.int64_t NAT = np.datetime64('nat').astype('int64')
    cdef long nanos = 10 ** 9 * 3600

    for j in range(nc):
        for i in range(nr):
            if a[i, j] == NAT:
                result[i, j] = NAT
            else:
                result[i, j] = <long> round(a[i, j] / nanos) * nanos
    return result.astype('datetime64[ns]')

def round_D(ndarray[np.float64_t, ndim=2] a):
    cdef int i, j
    cdef int nr = a.shape[0]
    cdef int nc = a.shape[1]
    cdef ndarray[np.int64_t, ndim=2] result = np.empty((nr, nc), dtype='int64')
    cdef np.int64_t NAT = np.datetime64('nat').astype('int64')
    cdef long nanos = 10 ** 9 * 3600 * 24

    for j in range(nc):
        for i in range(nr):
            if a[i, j] == NAT:
                result[i, j] = NAT
            else:
                result[i, j] = <long> round(a[i, j] / nanos) * nanos
    return result.astype('datetime64[ns]')

def round_Y(ndarray[np.float64_t, ndim=2] a):
    cdef int i, j
    cdef int nr = a.shape[0]
    cdef int nc = a.shape[1]
    cdef ndarray[np.float64_t, ndim=2] result = np.empty((nr, nc), dtype='float64')
    cdef np.int64_t NAT = np.datetime64('nat').astype('int64')
    cdef ndarray[np.int64_t] days_to_nano
    cdef int days_since_1600 = 135140
    cdef long year_nanos = 10 ** 9 * 86400
    cdef long cur_year_nanos, idx

    days_to_nano = create_days_to_nano_years()

    for j in range(nc):
        for i in range(nr):
            if a[i, j] == NAT:
                result[i, j] = NAT
            else:
                idx = <long> (a[i, j] // year_nanos) + days_since_1600
                cur_year_nanos = days_to_nano[idx]
                idx += 365
                while days_to_nano[idx] == cur_year_nanos:
                    idx += 1
                if a[i, j] - cur_year_nanos > days_to_nano[idx] - a[i, j]:
                    result[i, j] = days_to_nano[idx]
                else:
                    result[i, j] = cur_year_nanos
    return result.astype('datetime64[ns]')