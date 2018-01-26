import numpy as np
from . import utils
import dexplo._libs.validate_arrays as va


def _check_1d_arrays(a, b, kind):
    if kind == 'O':
        return va.is_equal_1d_object(a, b)
    return ((a == b) | (np.isnan(a) & np.isnan(b))).all()


def assert_frame_equal(df1, df2):
    if df1.shape != df2.shape:
        raise AssertionError('DataFrame shapes are not equal, '
                             f'{df1.shape} != {df2.shape}')

    for i, col in enumerate(df1.columns):
        if df2.columns[i] != col:
            raise AssertionError(f'column number {i} in left DataFrame not '
                                 f' {col} != {df2.columns[i]}')

        kind1, loc1, _ = df1._column_dtype[col].values
        arr1 = df1._data[kind1][:, loc1]

        kind2, loc2, order = df2._column_dtype[col].values
        arr2 = df2._data[kind2][:, loc2]

        if kind1 != kind2:
            dtype1 = utils.convert_kind_to_dtype(kind1)
            dtype2 = utils.convert_kind_to_dtype(kind2)
            raise AssertionError(f'The data types of column {col} are not '
                                 f'equal. {dtype1} != {dtype2}')

        if not _check_1d_arrays(arr1, arr2, kind1):
            raise AssertionError(f'The values of column {col} are not equal')
