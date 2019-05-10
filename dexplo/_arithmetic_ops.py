from collections import defaultdict
from . import _utils as utils
from ._libs import math_oper_string as mos


class OP_2D:
    '''
    Arithmetic and comparison operations with a DataFrame and another DataFrame/Array
    '''

    def __init__(self, left, right, string_op):
        self.left = left
        self.right = right
        self.string_op = string_op

        self.shapes_equal = left.shape == right.shape
        self.columns_equal = left.shape[1] == right.shape[1]
        self.rows_equal = left.shape[0] == right.shape[0]
        self.one_row = left.shape[0] == 1 or right.shape[0] == 1
        self.one_column = left.shape[1] == 1 or right.shape[1] == 1

        self.is_compatible_shape()

    def is_compatible_shape(self):
        if not self.shapes_equal:
            err1 = self.columns_equal and not self.one_row
            err2 = self.rows_equal and not self.one_column
            if err1 or err2:
                raise ValueError(f'Incompatible shapes: left {self.left.shape} vs '
                                 f'right {self.right.shape}. Shapes must be equal or have '
                                 f'same number of rows or columns with the other having one'
                                 f'row or one column')

    def operate(self):
        new_column_info = {}
        data_dict = defaultdict(list)
        str_map, str_reverse_map = {}, {}
        if self.shapes_equal:
            for i, (col1, col2) in enumerate(zip(self.left._columns, self.right._columns)):
                dtype1, loc1, order1 = self.left._column_info[col1].values
                dtype2, loc2, order2 = self.right._column_info[col2].values
                arr1 = self.left._data[dtype1][:, loc1]
                arr2 = self.right._data[dtype2][:, loc2]

                if dtype1 != 'S' and dtype2 != 'S':
                    try:
                        arr_final = getattr(arr1, self.string_op)(arr2)
                    except TypeError:
                        raise TypeError(f'Columns {col1} and {col2} have incompatible types. '
                                        f'{utils._DT[dtype1]} vs {utils._DT[dtype2]}')
                    new_kind = arr_final.dtype.kind
                    loc = len(data_dict[new_kind])
                elif dtype1 == 'S' and dtype2 == 'S':
                    srm1 = self.left._str_reverse_map[loc1]
                    srm2 = self.right._str_reverse_map[loc2]
                    func = getattr(mos, f'str{self.string_op}arr')
                    arr_final, cur_str_map, cur_str_reverse_map = func(arr1, arr2, srm1, srm2)
                    new_kind = 'S'
                    loc = len(data_dict[new_kind])
                    str_map[loc] = cur_str_map
                    str_reverse_map[loc] = cur_str_reverse_map

                new_column_info[col1] = utils.Column(new_kind, loc, i)
                data_dict[new_kind].append(arr_final)

        new_data = utils.concat_stat_arrays(data_dict)
        new_columns = self.left._columns.copy()
        return self.left._construct_from_new(new_data, new_column_info, new_columns, str_map, str_reverse_map)
