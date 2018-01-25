import pandas_lite as pd
import numpy as np
from numpy import array
import pytest
from pandas_lite.tests.testing import assert_frame_equal

df = pd.DataFrame({'a': [1, 2, 5, 9, 3, 4, 5, 1],
                   'b': [1.5, 8, 9, 1, 2, 3, 2, 8],
                   'c': list('abcdefgh'),
                   'd': [True, False, True, False] * 2,
                   'e': [10, 20, 30, 4, 5, 6, 7, 8],
                   'f': [1., 3, 3, 3, 11, 4, 5, 1],
                   'g': list('xyxxyyxy'),
                   'h': [3, 4, 5, 6, 7, 8, 9, 0]})


class TestFrameConstructor(object):

    def setup_method(self, method):

        self.df = pd.DataFrame({'a': [1, 2, 5, 9, 3, 4, 5, 1],
                                'b': [1.5, 8, 9, 1, 2, 3, 2, 8],
                                'c': list('abcdefgh'),
                                'd': [True, False, True, False] * 2,
                                'e': [10, 20, 30, 4, 5, 6, 7, 8],
                                'f': [1., 3, 3, 3, 11, 4, 5, 1],
                                'g': list('xyxxyyxy'),
                                'h': [3, 4, 5, 6, 7, 8, 9, 0]})


class TestScalarSelection:

    def test_scalar_selection(self):
        assert(df[5, -1] == 8)
        assert(df[3, 2] == 'd')
        assert(df[4, 'g'] == 'y')
        assert(df[1, 1] == 8)
        assert(df[3, 'h'] == 6)
        assert(df[0, 'e'] == 10)


class TestRowOnlySelection:

    def test_scalar_row_selection(self):
        # slice all
        df1 = df[:, :]
        assert_frame_equal(df, df1)

        # scalar row
        df1 = df[5, :]
        data = {'a': array([4]), 'b': array([3.]),
                'c': array(['f'], dtype='<U1'),
                'd': array([False], dtype=bool), 'e': array([6]),
                'f': array([4.]), 'g': array(['y'], dtype='<U1'),
                'h': array([8])}
        df2 = pd.DataFrame(data)
        assert_frame_equal(df1, df2)

        df1 = df[-1, :]
        data = {'a': [1], 'b': [8.0], 'c': ['h'], 'd': [False],
                'e': [8], 'f': [1.0], 'g': ['y'], 'h': [0]}
        df2 = pd.DataFrame(data)
        assert_frame_equal(df1, df2)

    def test_list_of_row_selection(self):
        df1 = df[[0, 4, 5], :]
        data = {'a': [1, 3, 4],
                'b': [1.5, 2.0, 3.0],
                'c': ['a', 'e', 'f'],
                'd': [True, True, False],
                'e': [10, 5, 6],
                'f': [1.0, 11.0, 4.0],
                'g': ['x', 'y', 'y'],
                'h': [3, 7, 8]}
        df2 = pd.DataFrame(data)
        assert_frame_equal(df1, df2)

        df1 = df[[-4], :]
        data = {'a': [3],
                'b': [2.0],
                'c': ['e'],
                'd': [True],
                'e': [5],
                'f': [11.0],
                'g': ['y'],
                'h': [7]}
        df2 = pd.DataFrame(data)
        assert_frame_equal(df1, df2)

    def test_slice_of_row_selection(self):
        df1 = df[2:6, :]
        data = {'a': [5, 9, 3, 4],
                'b': [9.0, 1.0, 2.0, 3.0],
                'c': ['c', 'd', 'e', 'f'],
                'd': [True, False, True, False],
                'e': [30, 4, 5, 6],
                'f': [3.0, 3.0, 11.0, 4.0],
                'g': ['x', 'x', 'y', 'y'],
                'h': [5, 6, 7, 8]}
        df2 = pd.DataFrame(data)
        assert_frame_equal(df1, df2)

        df1 = df[-3:, :]
        data = {'a': [4, 5, 1],
                'b': [3.0, 2.0, 8.0],
                'c': ['f', 'g', 'h'],
                'd': [False, True, False],
                'e': [6, 7, 8],
                'f': [4.0, 5.0, 1.0],
                'g': ['y', 'x', 'y'],
                'h': [8, 9, 0]}
        df2 = pd.DataFrame(data)
        assert_frame_equal(df1, df2)

        df1 = df[1:6:3, :]
        data = {'a': [2, 3],
                'b': [8.0, 2.0],
                'c': ['b', 'e'],
                'd': [False, True],
                'e': [20, 5],
                'f': [3.0, 11.0],
                'g': ['y', 'y'],
                'h': [4, 7]}
        df2 = pd.DataFrame(data)
        assert_frame_equal(df1, df2)


class TestColumnOnlySelection:

    def test_scalar_col_selection(self):
        df1 = df[:, 4]
        data = {'e': [10, 20, 30, 4, 5, 6, 7, 8]}
        df2 = pd.DataFrame(data)
        assert_frame_equal(df1, df2)

        df1 = df[:, -2]
        data = {'g': ['x', 'y', 'x', 'x', 'y', 'y', 'x', 'y']}
        df2 = pd.DataFrame(data)
        assert_frame_equal(df1, df2)

        df1 = df[:, 'd']
        data = {'d': [True, False, True, False, True, False, True, False]}
        df2 = pd.DataFrame(data)
        assert_frame_equal(df1, df2)

        with pytest.raises(ValueError):
            df[:, 'asdf']

    def test_list_of_integer_col_selection(self):
        df1 = df[:, [4, 6, 1]]
        data = {'b': [1.5, 8.0, 9.0, 1.0, 2.0, 3.0, 2.0, 8.0],
                'e': [10, 20, 30, 4, 5, 6, 7, 8],
                'g': ['x', 'y', 'x', 'x', 'y', 'y', 'x', 'y']}
        df2 = pd.DataFrame(data, columns=['e', 'g', 'b'])
        assert_frame_equal(df1, df2)

        df1 = df[:, [3]]
        data = {'d': [True, False, True, False, True, False, True, False]}
        df2 = pd.DataFrame(data)
        assert_frame_equal(df1, df2)

    def test_list_of_string_col_selection(self):
        df1 = df[:, ['b', 'd', 'a']]
        data = {'a': [1, 2, 5, 9, 3, 4, 5, 1],
                'b': [1.5, 8.0, 9.0, 1.0, 2.0, 3.0, 2.0, 8.0],
                'd': [True, False, True, False, True, False, True, False]}
        df2 = pd.DataFrame(data, columns=['b', 'd', 'a'])
        assert_frame_equal(df1, df2)

        df1 = df[:, ['a']]
        data = {'a': [1, 2, 5, 9, 3, 4, 5, 1]}
        df2 = pd.DataFrame(data)
        assert_frame_equal(df1, df2)

    def test_list_of_string_and_integer_col_selection(self):
        df1 = df[:, ['b', 5]]
        data = {'b': [1.5, 8.0, 9.0, 1.0, 2.0, 3.0, 2.0, 8.0],
                'f': [1.0, 3.0, 3.0, 3.0, 11.0, 4.0, 5.0, 1.0]}
        df2 = pd.DataFrame(data, columns=['b', 'f'])
        assert_frame_equal(df1, df2)

        df1 = df[:, [-2, 'c', 0, 'd']]
        data = {'a': [1, 2, 5, 9, 3, 4, 5, 1],
                'c': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'],
                'd': [True, False, True, False, True, False, True, False],
                'g': ['x', 'y', 'x', 'x', 'y', 'y', 'x', 'y']}
        df2 = pd.DataFrame(data, columns=['g', 'c', 'a', 'd'])
        assert_frame_equal(df1, df2)

        with pytest.raises(ValueError):
            df[:, ['b', 5, 'e', 'f']]

    def test_slice_with_integers_col_selection(self):
        df1 = df[:, 3:6]
        data = {'d': [True, False, True, False, True, False, True, False],
                'e': [10, 20, 30, 4, 5, 6, 7, 8],
                'f': [1.0, 3.0, 3.0, 3.0, 11.0, 4.0, 5.0, 1.0]}
        df2 = pd.DataFrame(data, columns=['d', 'e', 'f'])
        assert_frame_equal(df1, df2)

        df1 = df[:, -4::2]
        data = {'e': [10, 20, 30, 4, 5, 6, 7, 8],
                'g': ['x', 'y', 'x', 'x', 'y', 'y', 'x', 'y']}
        df2 = pd.DataFrame(data, columns=['e', 'g'])
        assert_frame_equal(df1, df2)

    def test_slice_with_labels_col_selection(self):
        df1 = df[:, 'c':'f']
        data = {'c': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'],
                'd': [True, False, True, False, True, False, True, False],
                'e': [10, 20, 30, 4, 5, 6, 7, 8],
                'f': [1.0, 3.0, 3.0, 3.0, 11.0, 4.0, 5.0, 1.0]}
        df2 = pd.DataFrame(data, columns=['c', 'd', 'e', 'f'])
        assert_frame_equal(df1, df2)

        df1 = df[:, :'b']
        data = {'a': [1, 2, 5, 9, 3, 4, 5, 1],
                'b': [1.5, 8.0, 9.0, 1.0, 2.0, 3.0, 2.0, 8.0]}
        df2 = pd.DataFrame(data, columns=['a', 'b'])
        assert_frame_equal(df1, df2)

        df1 = df[:, 'g':'b':-2]
        data = {'c': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'],
                'e': [10, 20, 30, 4, 5, 6, 7, 8],
                'g': ['x', 'y', 'x', 'x', 'y', 'y', 'x', 'y']}
        df2 = pd.DataFrame(data, columns=['g', 'e', 'c'])
        assert_frame_equal(df1, df2)

    def test_slice_labels_and_integer_col_selection(self):
        df1 = df[:, 'c':5]
        data = {'c': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'],
                'd': [True, False, True, False, True, False, True, False],
                'e': [10, 20, 30, 4, 5, 6, 7, 8]}
        df2 = pd.DataFrame(data, columns=['c', 'd', 'e'])
        assert_frame_equal(df1, df2)

        df1 = df[:, 6:'d':-1]
        data = {'d': [True, False, True, False, True, False, True, False],
                'e': [10, 20, 30, 4, 5, 6, 7, 8],
                'f': [1.0, 3.0, 3.0, 3.0, 11.0, 4.0, 5.0, 1.0],
                'g': ['x', 'y', 'x', 'x', 'y', 'y', 'x', 'y']}
        df2 = pd.DataFrame(data, columns=['g', 'f', 'e', 'd'])
        assert_frame_equal(df1, df2)


class TestSimultaneousRowColumnSelection:

    def test_scalar_row_with_list_slice_column_selection(self):
        df1 = df[3, [4, 5, 6]]
        data = {'e': [4], 'f': [3.0], 'g': ['x']}
        df2 = pd.DataFrame(data, columns=['e', 'f', 'g'])
        assert_frame_equal(df1, df2)

        df1 = df[1, [-1]]
        data = {'h': [4]}
        df2 = pd.DataFrame(data)
        assert_frame_equal(df1, df2)

        df1 = df[0, ['g', 'd']]
        data = {'d': [True], 'g': ['x']}
        df2 = pd.DataFrame(data, columns=['g', 'd'])
        assert_frame_equal(df1, df2)

        df1 = df[0, ['d']]
        data = {'d': [True]}
        df2 = pd.DataFrame(data)
        assert_frame_equal(df1, df2)

        df1 = df[-2, 2:6]
        data = {'c': ['g'], 'd': [True], 'e': [7], 'f': [5.0]}
        df2 = pd.DataFrame(data, columns=['c', 'd', 'e', 'f'])
        assert_frame_equal(df1, df2)

        df1 = df[4, 'f':'b':-1]
        data = {'b': [2.0], 'c': ['e'], 'd': [True], 'e': [5], 'f': [11.0]}
        df2 = pd.DataFrame(data, columns=['f', 'e', 'd', 'c', 'b'])
        assert_frame_equal(df1, df2)

    def test_scalar_column_with_list_slice_row_selection(self):
        df1 = df[[4, 6], 2]
        data = {'c': ['e', 'g']}
        df2 = pd.DataFrame(data)
        assert_frame_equal(df1, df2)

        df1 = df[[4], 2]
        data = {'c': ['e']}
        df2 = pd.DataFrame(data)
        assert_frame_equal(df1, df2)

        df1 = df[[5, 2], 'f']
        data = {'f': [4.0, 3.0]}
        df2 = pd.DataFrame(data)
        assert_frame_equal(df1, df2)

        df1 = df[3:, 'f']
        data = {'f': [3.0, 11.0, 4.0, 5.0, 1.0]}
        df2 = pd.DataFrame(data)
        assert_frame_equal(df1, df2)

        df1 = df[5::-2, 'b']
        data = {'b': [3.0, 1.0, 8.0]}
        df2 = pd.DataFrame(data)
        assert_frame_equal(df1, df2)

    def test_list_slice_row_with_list_slice_column_selection(self):
        df1 = df[[3, 4], [0, 6]]
        data = {'a': [9, 3], 'g': ['x', 'y']}
        df2 = pd.DataFrame(data, columns=['a', 'g'])
        assert_frame_equal(df1, df2)

        df1 = df[3::3, [6, 3, 1, 5]]
        data = {'b': [1.0, 2.0], 'd': [False, True],
                'f': [3.0, 5.0], 'g': ['x', 'x']}
        df2 = pd.DataFrame(data, columns=['g', 'd', 'b', 'f'])
        assert_frame_equal(df1, df2)

        df1 = df[3:, 'c':]
        data = {'c': ['d', 'e', 'f', 'g', 'h'],
                'd': [False, True, False, True, False],
                'e': [4, 5, 6, 7, 8],
                'f': [3.0, 11.0, 4.0, 5.0, 1.0],
                'g': ['x', 'y', 'y', 'x', 'y'],
                'h': [6, 7, 8, 9, 0]}
        df2 = pd.DataFrame(data, columns=['c', 'd', 'e', 'f', 'g', 'h'])
        assert_frame_equal(df1, df2)
        
