import numpy as np
cimport numpy as np
from numpy cimport ndarray
from numpy import nan

cdef extern from "numpy/npy_math.h":
    bint npy_isnan(double x)

def to_csv(ndarray[object, ndim=2] a, ndarray[object] columns, ndarray[np.int64_t] dtypes,
           str fn, str sep):
    cdef:
        Py_ssize_t nr = a.shape[0], nc = a.shape[1] - 1
    with open(fn, 'w') as f:
        for j in range(nc):
            f.write(f'{columns[j]}{sep}')
        f.write(f'{columns[nc]}\n')

        # 0: float, 1: str, 2: datetime/timedelta, 3: int/bool

        for i in range(nr):
            for j in range(nc):
                if dtypes[j] == 0:
                    if not npy_isnan(a[i, j]):
                        f.write(f'{a[i, j]}')
                elif dtypes[j] == 1:
                    if a[i, j] is not None:
                        f.write(a[i, j])
                elif dtypes[j] == 2:
                    if not npy_isnan(a[i, j]):
                        f.write(f'{a[i, j]}')
                else:
                    f.write(f'{a[i, j]}')
                f.write(sep)

            if dtypes[nc] == 0:
                if not npy_isnan(a[i, nc]):
                    f.write(f'{a[i, nc]}')
            elif dtypes[nc] == 1:
                if a[i, nc] is not None:
                    f.write(a[i, nc])
            elif dtypes[nc] == 2:
                if not npy_isnan(a[i, nc]):
                    f.write(f'{a[i, nc]}')
            else:
                f.write(f'{a[i, nc]}')
            f.write('\n')