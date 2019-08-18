import numpy as np
from numpy import ndarray
import dexplo._utils as utils
from dexplo._frame import DataFrame
import dexplo._libs.validate_arrays as va


def _check_1d_arrays(a: ndarray, b: ndarray, kind: str, tol: float = 10 ** -4) -> bool:
    if kind == 'O':
        if not va.is_equal_1d_object(a, b):
            raise AssertionError(f'The values of the columns are not equal')
        return True
    elif kind == 'f':
        with np.errstate(invalid='ignore'):
            criteria1 = np.abs(a - b) < tol
            criteria2 = np.isnan(a) & np.isnan(b)
            criteria3 = np.isinf(a) & np.isinf(b)
        return (criteria1 | criteria2 | criteria3).all()
    else:
        try:
            np.testing.assert_array_equal(a, b)
        except AssertionError:
            return False
        return True


def assert_frame_equal(df1: DataFrame, df2: DataFrame) -> None:
    if df1.shape != df2.shape:
        raise AssertionError('DataFrame shapes are not equal, '
                             f'{df1.shape} != {df2.shape}')

    for i, col in enumerate(df1.columns):
        if df2.columns[i] != col:
            raise AssertionError(f'column number {i} in left DataFrame not equal to right '
                                 f'{col} != {df2.columns[i]}')

        kind1, loc1 = df1._get_col_dtype_loc(col)  # type: str, int
        arr1: ndarray = df1._data[kind1][:, loc1]

        kind2, loc2 = df2._get_col_dtype_loc(col)  # type: str, int
        arr2: ndarray = df2._data[kind2][:, loc2]

        if kind1 != kind2:
            dtype1 = utils.convert_kind_to_dtype(kind1)
            dtype2 = utils.convert_kind_to_dtype(kind2)
            raise AssertionError(f'The data types of column {col} are not '
                                 f'equal. {dtype1} != {dtype2}')

        if kind1 == 'S':
            srm1 = df1._str_reverse_map[loc1]
            srm2 = df2._str_reverse_map[loc2]
            if not va.is_equal_str_cat_array(arr1, arr2, srm1, srm2):
                raise AssertionError(f'The values of column {col} are not equal')
        elif not _check_1d_arrays(arr1, arr2, kind1):
            raise AssertionError(f'The values of column {col} are not equal')


def assert_array_equal(arr1, arr2, check_dtype=True):
    if arr1.shape != arr2.shape:
        raise AssertionError(f'Array shapes not equal: {arr1.shape} != {arr2.shape}')

    if check_dtype and arr1.dtype.kind != arr2.dtype.kind:
        raise AssertionError(f'Array data types not equal: {arr1.dtype} != {arr2.dtype}')

    if arr1.ndim == 1:
        if not _check_1d_arrays(arr1, arr2, arr1.dtype.kind):
            raise AssertionError('Arrays not equal')
    else:
        for i in range(arr1.shape[1]):
            if not _check_1d_arrays(arr1[:, i], arr2[:, i], arr1.dtype.kind):
                raise AssertionError(f'Column {i} not equal')


def assert_dict_list(d1, d2):
    for key, values1 in d1.items():
        for v1, v2 in zip(values1, d2[key]):
            if v1 != v2:
                if isinstance(v1, (float, np.floating)) and np.isnan(v1) and \
                   isinstance(v2, (float, np.floating)) and np.isnan(v2):
                    continue
                raise AssertionError('Lists are not equal')
