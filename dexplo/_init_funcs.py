from collections import defaultdict
from typing import Union, Dict, List, Optional, Tuple, Set

import numpy as np
from numpy import ndarray

from ._libs import validate_arrays as va
from . import _utils as utils


DataC = Union[Dict[str, Union[ndarray, List]], ndarray]
DictListArr = Dict[str, List[ndarray]]
ColumnT = Optional[Union[List[str], ndarray]]
ColInfoT = Dict[str, utils.Column]


def columns_from_dict(columns: ColumnT, data: DataC) -> ndarray:
    """
    Sets the column names when a dictionary is passed to the DataFrame constructor.
    If the columns parameter is not none, its elements must match the dictionary keys.

    Parameters
    ----------
    columns: List or array of strings of column names
    data: Dictionary of lists or 1d arrays

    Returns
    -------
    None

    """
    if columns is None:
        columns = np.array(list(data))
    columns = check_column_validity(columns)
    if set(columns) != set(data.keys()):
        raise ValueError("Column names don't match dictionary keys")
    return columns


def columns_from_array(columns: ColumnT, num_cols: int) -> ndarray:
    """
    When an array or list is passed to the `columns` parameter in the DataFrame constructor

    Parameters
    ----------
    columns : List or array of strings column names
    num_cols : Integer of the number of columns

    Returns
    -------
    None

    """
    if columns is None:
        col_list: List[str] = [f'a{str(i)}' for i in range(num_cols)]
        columns: ndarray = np.array(col_list, dtype='O')
    else:
        columns = check_column_validity(columns)
        if len(columns) != num_cols:
            raise ValueError(f'Number of column names {len(columns)} does not equal '
                             f'number of columns of data array {num_cols}')
    return columns


def check_column_validity(cols: ColumnT) -> ndarray:
    """
    Determine if column names are valid
    Parameters
    ----------
    cols : list or array of strings

    Returns
    -------
    Nothing when valid and raises an error if duplicated or non-string
    """
    if not isinstance(cols, (list, ndarray)):
        raise TypeError('Columns must be a list or an array')
    if isinstance(cols, ndarray):
        cols = utils.try_to_squeeze_array(cols)

    col_set: Set[str] = set()
    for i, col in enumerate(cols):
        if not isinstance(col, str):
            raise TypeError('Column names must be a string')
        if col in col_set:
            raise ValueError(f'Column name {col} is duplicated. Column names must be unique')
        col_set.add(col)
    return np.asarray(cols, dtype='O')


def data_from_dict(data: DataC) -> None:
    """
    Sets the _data attribute whenever a dictionary is passed to the `data` parameter in the
    DataFrame constructor. Also sets `_column_info`

    Parameters
    ----------
    data: Dictionary of lists or 1d arrays

    Returns
    -------
    None
    """
    column_info: ColInfoT = {}
    data_dict: DictListArr = defaultdict(list)
    for i, (col, values) in enumerate(data.items()):
        if isinstance(values, list):
            arr: ndarray = utils.convert_list_to_single_arr(values)
        elif isinstance(values, ndarray):
            arr = values
        else:
            raise TypeError('Values of dictionary must be an array or a list')
        arr = utils.maybe_convert_1d_array(arr, col)
        kind: str = arr.dtype.kind
        loc: int = len(data_dict.get(kind, []))
        data_dict[kind].append(arr)
        column_info[col] = utils.Column(kind, loc, i)

        if i == 0:
            first_len: int = len(arr)
        elif len(arr) != first_len:
            raise ValueError('All columns must be the same length')

    return concat_arrays(data_dict), column_info


def data_from_array(data: ndarray, columns: ndarray) -> Tuple:
    if data.dtype.kind == 'O':
        return data_from_object_array(data, columns)
    else:
        return data_from_typed_array(data, columns)


def data_from_typed_array(data: ndarray, columns: ndarray) -> Tuple:
    """
    Stores entire array, `data` into `self._data` as one kind

    Parameters
    ----------
    data : A homogeneous array
    columns: Array

    Returns
    -------
    None
    """
    kind: str = data.dtype.kind
    if kind == 'U':
        data = data.astype('O')
    elif kind == 'M':
        data = data.astype('datetime64[ns]')
    elif kind == 'm':
        data = data.astype('timedelta64[ns]')

    if data.ndim == 1:
        data = data[:, np.newaxis]

    kind = data.dtype.kind
    # Force array to be fortran ordered
    new_data = {kind: np.asfortranarray(data)}
    column_info: ColInfoT = {col: utils.Column(kind, i, i) for i, col in enumerate(columns)}
    return new_data, column_info


def data_from_object_array(data: ndarray, columns: ndarray) -> Tuple:
    """
    Special initialization when array if of kind 'O'. Must check each column individually

    Parameters
    ----------
    data : A numpy object array

    Returns
    -------
    None
    """
    if data.ndim == 1:
        data = data[:, np.newaxis]

    column_info: ColInfoT = {}
    data_dict: DictListArr = defaultdict(list)
    for i, col in enumerate(columns):
        arr: ndarray = va.maybe_convert_object_array(data[:, i], col)
        kind: str = arr.dtype.kind
        loc: int = len(data_dict[kind])
        data_dict[kind].append(arr)
        column_info[col] = utils.Column(kind, loc, i)

    return concat_arrays(data_dict), column_info


def concat_arrays(data_dict: DictListArr) -> Dict[str, ndarray]:
    """
    Concatenates the lists for each kind into a single array
    """
    new_data: Dict[str, ndarray] = {}
    for dtype, arrs in data_dict.items():
        if arrs:
            if len(arrs) == 1:
                new_data[dtype] = arrs[0].reshape(-1, 1)
            else:
                arrs = np.column_stack(arrs)
                new_data[dtype] = np.asfortranarray(arrs)
    return new_data
