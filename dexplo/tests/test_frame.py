import dexplo as de
import numpy as np
from numpy import array, nan
import pytest
from dexplo.testing import assert_frame_equal

df = de.DataFrame({'a': [1, 2, 5, 9, 3, 4, 5, 1],
                   'b': [1.5, 8, 9, 1, 2, 3, 2, 8],
                   'c': list('abcdefgh'),
                   'd': [True, False, True, False] * 2,
                   'e': [10, 20, 30, 4, 5, 6, 7, 8],
                   'f': [1., 3, 3, 3, 11, 4, 5, 1],
                   'g': list('xyxxyyxy'),
                   'h': [3, 4, 5, 6, 7, 8, 9, 0]},
                  columns=list('abcdefgh'))


class TestFrameConstructor(object):

    def setup_method(self, method):
        self.df = de.DataFrame({'a': [1, 2, 5, 9, 3, 4, 5, 1],
                                'b': [1.5, 8, 9, 1, 2, 3, 2, 8],
                                'c': list('abcdefgh'),
                                'd': [True, False, True, False] * 2,
                                'e': [10, 20, 30, 4, 5, 6, 7, 8],
                                'f': [1., 3, 3, 3, 11, 4, 5, 1],
                                'g': list('xyxxyyxy'),
                                'h': [3, 4, 5, 6, 7, 8, 9, 0]})


class TestScalarSelection:

    def test_scalar_selection(self):
        assert (df[5, -1] == 8)
        assert (df[3, 2] == 'd')
        assert (df[4, 'g'] == 'y')
        assert (df[1, 1] == 8)
        assert (df[3, 'h'] == 6)
        assert (df[0, 'e'] == 10)


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
        df2 = de.DataFrame(data)
        assert_frame_equal(df1, df2)

        df1 = df[-1, :]
        data = {'a': [1], 'b': [8.0], 'c': ['h'], 'd': [False],
                'e': [8], 'f': [1.0], 'g': ['y'], 'h': [0]}
        df2 = de.DataFrame(data)
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
        df2 = de.DataFrame(data)
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
        df2 = de.DataFrame(data)
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
        df2 = de.DataFrame(data)
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
        df2 = de.DataFrame(data)
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
        df2 = de.DataFrame(data)
        assert_frame_equal(df1, df2)


class TestColumnOnlySelection:

    def test_scalar_col_selection(self):
        df1 = df[:, 4]
        data = {'e': [10, 20, 30, 4, 5, 6, 7, 8]}
        df2 = de.DataFrame(data)
        assert_frame_equal(df1, df2)

        df1 = df[:, -2]
        data = {'g': ['x', 'y', 'x', 'x', 'y', 'y', 'x', 'y']}
        df2 = de.DataFrame(data)
        assert_frame_equal(df1, df2)

        df1 = df[:, 'd']
        data = {'d': [True, False, True, False, True, False, True, False]}
        df2 = de.DataFrame(data)
        assert_frame_equal(df1, df2)

        with pytest.raises(KeyError):
            df[:, 'asdf']

    def test_list_of_integer_col_selection(self):
        df1 = df[:, [4, 6, 1]]
        data = {'b': [1.5, 8.0, 9.0, 1.0, 2.0, 3.0, 2.0, 8.0],
                'e': [10, 20, 30, 4, 5, 6, 7, 8],
                'g': ['x', 'y', 'x', 'x', 'y', 'y', 'x', 'y']}
        df2 = de.DataFrame(data, columns=['e', 'g', 'b'])
        assert_frame_equal(df1, df2)

        df1 = df[:, [3]]
        data = {'d': [True, False, True, False, True, False, True, False]}
        df2 = de.DataFrame(data)
        assert_frame_equal(df1, df2)

    def test_list_of_string_col_selection(self):
        df1 = df[:, ['b', 'd', 'a']]
        data = {'a': [1, 2, 5, 9, 3, 4, 5, 1],
                'b': [1.5, 8.0, 9.0, 1.0, 2.0, 3.0, 2.0, 8.0],
                'd': [True, False, True, False, True, False, True, False]}
        df2 = de.DataFrame(data, columns=['b', 'd', 'a'])
        assert_frame_equal(df1, df2)

        df1 = df[:, ['a']]
        data = {'a': [1, 2, 5, 9, 3, 4, 5, 1]}
        df2 = de.DataFrame(data)
        assert_frame_equal(df1, df2)

    def test_list_of_string_and_integer_col_selection(self):
        df1 = df[:, ['b', 5]]
        data = {'b': [1.5, 8.0, 9.0, 1.0, 2.0, 3.0, 2.0, 8.0],
                'f': [1.0, 3.0, 3.0, 3.0, 11.0, 4.0, 5.0, 1.0]}
        df2 = de.DataFrame(data, columns=['b', 'f'])
        assert_frame_equal(df1, df2)

        df1 = df[:, [-2, 'c', 0, 'd']]
        data = {'a': [1, 2, 5, 9, 3, 4, 5, 1],
                'c': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'],
                'd': [True, False, True, False, True, False, True, False],
                'g': ['x', 'y', 'x', 'x', 'y', 'y', 'x', 'y']}
        df2 = de.DataFrame(data, columns=['g', 'c', 'a', 'd'])
        assert_frame_equal(df1, df2)

        with pytest.raises(ValueError):
            df[:, ['b', 5, 'e', 'f']]

    def test_slice_with_integers_col_selection(self):
        df1 = df[:, 3:6]
        data = {'d': [True, False, True, False, True, False, True, False],
                'e': [10, 20, 30, 4, 5, 6, 7, 8],
                'f': [1.0, 3.0, 3.0, 3.0, 11.0, 4.0, 5.0, 1.0]}
        df2 = de.DataFrame(data, columns=['d', 'e', 'f'])
        assert_frame_equal(df1, df2)

        df1 = df[:, -4::2]
        data = {'e': [10, 20, 30, 4, 5, 6, 7, 8],
                'g': ['x', 'y', 'x', 'x', 'y', 'y', 'x', 'y']}
        df2 = de.DataFrame(data, columns=['e', 'g'])
        assert_frame_equal(df1, df2)

    def test_slice_with_labels_col_selection(self):
        df1 = df[:, 'c':'f']
        data = {'c': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'],
                'd': [True, False, True, False, True, False, True, False],
                'e': [10, 20, 30, 4, 5, 6, 7, 8],
                'f': [1.0, 3.0, 3.0, 3.0, 11.0, 4.0, 5.0, 1.0]}
        df2 = de.DataFrame(data, columns=['c', 'd', 'e', 'f'])
        assert_frame_equal(df1, df2)

        df1 = df[:, :'b']
        data = {'a': [1, 2, 5, 9, 3, 4, 5, 1],
                'b': [1.5, 8.0, 9.0, 1.0, 2.0, 3.0, 2.0, 8.0]}
        df2 = de.DataFrame(data, columns=['a', 'b'])
        assert_frame_equal(df1, df2)

        df1 = df[:, 'g':'b':-2]
        data = {'c': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'],
                'e': [10, 20, 30, 4, 5, 6, 7, 8],
                'g': ['x', 'y', 'x', 'x', 'y', 'y', 'x', 'y']}
        df2 = de.DataFrame(data, columns=['g', 'e', 'c'])
        assert_frame_equal(df1, df2)

    def test_slice_labels_and_integer_col_selection(self):
        df1 = df[:, 'c':5]
        data = {'c': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'],
                'd': [True, False, True, False, True, False, True, False],
                'e': [10, 20, 30, 4, 5, 6, 7, 8]}
        df2 = de.DataFrame(data, columns=['c', 'd', 'e'])
        assert_frame_equal(df1, df2)

        df1 = df[:, 6:'d':-1]
        data = {'d': [True, False, True, False, True, False, True, False],
                'e': [10, 20, 30, 4, 5, 6, 7, 8],
                'f': [1.0, 3.0, 3.0, 3.0, 11.0, 4.0, 5.0, 1.0],
                'g': ['x', 'y', 'x', 'x', 'y', 'y', 'x', 'y']}
        df2 = de.DataFrame(data, columns=['g', 'f', 'e', 'd'])
        assert_frame_equal(df1, df2)


class TestSimultaneousRowColumnSelection:

    def test_scalar_row_with_list_slice_column_selection(self):
        df1 = df[3, [4, 5, 6]]
        data = {'e': [4], 'f': [3.0], 'g': ['x']}
        df2 = de.DataFrame(data, columns=['e', 'f', 'g'])
        assert_frame_equal(df1, df2)

        df1 = df[1, [-1]]
        data = {'h': [4]}
        df2 = de.DataFrame(data)
        assert_frame_equal(df1, df2)

        df1 = df[0, ['g', 'd']]
        data = {'d': [True], 'g': ['x']}
        df2 = de.DataFrame(data, columns=['g', 'd'])
        assert_frame_equal(df1, df2)

        df1 = df[0, ['d']]
        data = {'d': [True]}
        df2 = de.DataFrame(data)
        assert_frame_equal(df1, df2)

        df1 = df[-2, 2:6]
        data = {'c': ['g'], 'd': [True], 'e': [7], 'f': [5.0]}
        df2 = de.DataFrame(data, columns=['c', 'd', 'e', 'f'])
        assert_frame_equal(df1, df2)

        df1 = df[4, 'f':'b':-1]
        data = {'b': [2.0], 'c': ['e'], 'd': [True], 'e': [5], 'f': [11.0]}
        df2 = de.DataFrame(data, columns=['f', 'e', 'd', 'c', 'b'])
        assert_frame_equal(df1, df2)

    def test_scalar_column_with_list_slice_row_selection(self):
        df1 = df[[4, 6], 2]
        data = {'c': ['e', 'g']}
        df2 = de.DataFrame(data)
        assert_frame_equal(df1, df2)

        df1 = df[[4], 2]
        data = {'c': ['e']}
        df2 = de.DataFrame(data)
        assert_frame_equal(df1, df2)

        df1 = df[[5, 2], 'f']
        data = {'f': [4.0, 3.0]}
        df2 = de.DataFrame(data)
        assert_frame_equal(df1, df2)

        df1 = df[3:, 'f']
        data = {'f': [3.0, 11.0, 4.0, 5.0, 1.0]}
        df2 = de.DataFrame(data)
        assert_frame_equal(df1, df2)

        df1 = df[5::-2, 'b']
        data = {'b': [3.0, 1.0, 8.0]}
        df2 = de.DataFrame(data)
        assert_frame_equal(df1, df2)

    def test_list_slice_row_with_list_slice_column_selection(self):
        df1 = df[[3, 4], [0, 6]]
        data = {'a': [9, 3], 'g': ['x', 'y']}
        df2 = de.DataFrame(data, columns=['a', 'g'])
        assert_frame_equal(df1, df2)

        df1 = df[3::3, [6, 3, 1, 5]]
        data = {'b': [1.0, 2.0], 'd': [False, True],
                'f': [3.0, 5.0], 'g': ['x', 'x']}
        df2 = de.DataFrame(data, columns=['g', 'd', 'b', 'f'])
        assert_frame_equal(df1, df2)

        df1 = df[3:, 'c':]
        data = {'c': ['d', 'e', 'f', 'g', 'h'],
                'd': [False, True, False, True, False],
                'e': [4, 5, 6, 7, 8],
                'f': [3.0, 11.0, 4.0, 5.0, 1.0],
                'g': ['x', 'y', 'y', 'x', 'y'],
                'h': [6, 7, 8, 9, 0]}
        df2 = de.DataFrame(data, columns=['c', 'd', 'e', 'f', 'g', 'h'])
        assert_frame_equal(df1, df2)


class TestBooleanSelection:

    df = de.DataFrame({'a': [0, 0, 5, 9, 3, 4, 5, 1],
                       'b': [0, 1.512344353, 8, 9, nan, 3, 2, 8],
                       'c': [''] + list('bgggzgh'),
                       'd': [False, False, True, False] * 2,
                       'e': [0, 20, 30, 4, 5, 6, 7, 8],
                       'f': [0., 3, 3, 3, 11, 4, 5, 1],
                       'g': ['', None, 'ad', 'effd', 'ef', None, 'ett', 'zzzz'],
                       'h': [0, 4, 5, 6, 7, 8, 9, 0],
                       'i': np.array([0, 7, 6, 5, 4, 3, 2, 11]),
                       'j': np.zeros(8, dtype='int'),
                       'k': np.ones(8) - 1,
                       'l': [nan] * 8},
                      columns=list('abcdefghijkl'))

    def test_integer_condition(self):
        criteria = self.df[:, 'a'] > 4
        df1 = self.df[criteria, :]
        df2 = self.df[[2, 3, 6], :]
        assert_frame_equal(df1, df2)

        criteria = self.df[:, 'a'] == 0
        df1 = self.df[criteria, :]
        df2 = self.df[[0, 1], :]
        assert_frame_equal(df1, df2)

        criteria = (self.df[:, 'a'] > 2) & (self.df[:, 'i'] < 6)
        df1 = self.df[criteria, :]
        df2 = self.df[[3, 4, 5, 6], :]
        assert_frame_equal(df1, df2)

        criteria = (self.df[:, 'a'] > 2) | (self.df[:, 'i'] < 6)
        df1 = self.df[criteria, :]
        df2 = self.df[[0, 2, 3, 4, 5, 6], :]
        assert_frame_equal(df1, df2)

        criteria = ~((self.df[:, 'a'] > 2) | (self.df[:, 'i'] < 6))
        df1 = self.df[criteria, :]
        df2 = self.df[[1, 7], :]
        assert_frame_equal(df1, df2)

        criteria = ~((self.df[:, 'a'] > 2) | (self.df[:, 'i'] < 6))
        df1 = self.df[criteria, ['d', 'b']]
        df2 = de.DataFrame({'b': [1.512344353, 8],
                            'd': [False, False]}, columns=['d', 'b'])
        assert_frame_equal(df1, df2)

    def test_list_of_booleans(self):
        criteria = [False, True, False, True, False, True, False, True]
        df1 = self.df[criteria, :]
        df2 = self.df[[1, 3, 5, 7], :]
        assert_frame_equal(df1, df2)

        criteria = [False, True, False, True, False, True, False]
        with pytest.raises(ValueError):
            self.df[criteria, :]

        criteria = [False, True, False, True, False, True] * 2
        df1 = self.df[:, criteria]
        df2 = self.df[:, list(range(1, 12, 2))]
        assert_frame_equal(df1, df2)

        criteria_row = [False, False, True, False, True, True, False, False]
        criteria_col = [False, True, False, True, False, True] * 2
        df1 = self.df[criteria_row, criteria_col]
        df2 = self.df[[2, 4, 5], list(range(1, 12, 2))]
        assert_frame_equal(df1, df2)

        with pytest.raises(TypeError):
            self.df[[0, 5], [False, 5]]

        with pytest.raises(ValueError):
            self.df[:, [True, False, True, False]]

        df1 = self.df[self.df[:, 'c'] == 'g', ['d', 'j']]
        df2 = de.DataFrame({'d': [True, False, False, True],
                            'j': [0, 0, 0, 0]}, columns=['d', 'j'])
        assert_frame_equal(df1, df2)

        with np.errstate(invalid='ignore'):
            df1 = self.df[self.df[:, 'b'] < 2, 'b']
            df2 = de.DataFrame({'b': [0, 1.512344353]})
            assert_frame_equal(df1, df2)



class TestSetItem:
    df = de.DataFrame({'a': [1, 5], 'b': ['eleni', 'teddy'], 'c': [nan, 5.4],
                       'd': [True, False]})

    df1 = de.DataFrame({'a': [1, 5, 7, 11], 'b': ['eleni', 'teddy', 'niko', 'penny'],
                        'c': [nan, 5.4, -1.1, .045], 'd': [True, False, False, True]})

    def test_setitem_scalar(self):
        df1 = self.df.copy()
        df1[0, 0] = -99
        df2 = de.DataFrame({'a': [-99, 5], 'b': ['eleni', 'teddy'], 'c': [nan, 5.4],
                            'd': [True, False]})
        assert_frame_equal(df1, df2)

        df1[0, 'b'] = 'pen'
        df2 = de.DataFrame({'a': [-99, 5], 'b': ['pen', 'teddy'], 'c': [nan, 5.4],
                            'd': [True, False]})
        assert_frame_equal(df1, df2)

        df1[1, 'b'] = nan
        df2 = de.DataFrame({'a': [-99, 5], 'b': ['pen', nan], 'c': [nan, 5.4],
                            'd': [True, False]})
        assert_frame_equal(df1, df2)

        with pytest.raises(TypeError):
            df1 = self.df.copy()
            df1[0, 0] = 'sfa'

        df1 = self.df.copy()
        df1[0, 'c'] = 4.3
        df2 = de.DataFrame({'a': [1, 5], 'b': ['eleni', 'teddy'], 'c': [4.3, 5.4],
                           'd': [True, False]})
        assert_frame_equal(df1, df2)

        df1 = self.df.copy()
        df1[0, 'a'] = nan
        df2 = de.DataFrame({'a': [nan, 5], 'b': ['eleni', 'teddy'], 'c': [nan, 5.4],
                            'd': [True, False]})
        assert_frame_equal(df1, df2)

        df1 = self.df.copy()
        df1[1, 'a'] = -9.9
        df2 = de.DataFrame({'a': [1, -9.9], 'b': ['eleni', 'teddy'], 'c': [nan, 5.4],
                            'd': [True, False]})
        assert_frame_equal(df1, df2)

    def test_setitem_entire_column_one_value(self):
        df1 = self.df.copy()
        df1[:, 'e'] = 5
        df2 = de.DataFrame({'a': [1, 5], 'b': ['eleni', 'teddy'], 'c': [nan, 5.4],
                            'd': [True, False], 'e': [5, 5]})
        assert_frame_equal(df1, df2)

        df1 = self.df.copy()
        df1[:, 'e'] = nan
        df2 = de.DataFrame({'a': [1, 5], 'b': ['eleni', 'teddy'], 'c': [nan, 5.4],
                            'd': [True, False], 'e': [nan, nan]})
        assert_frame_equal(df1, df2)

        df1 = self.df.copy()
        df1[:, 'e'] = 'grasshopper'
        df2 = de.DataFrame({'a': [1, 5], 'b': ['eleni', 'teddy'], 'c': [nan, 5.4],
                            'd': [True, False], 'e': ['grasshopper', 'grasshopper']})
        assert_frame_equal(df1, df2)

        df1 = self.df.copy()
        df1[:, 'e'] = True
        df2 = de.DataFrame({'a': [1, 5], 'b': ['eleni', 'teddy'], 'c': [nan, 5.4],
                            'd': [True, False], 'e': [True, True]})
        assert_frame_equal(df1, df2)

    def test_setitem_entire_new_colunm_from_array(self):
        df1 = self.df.copy()
        df1[:, 'e'] = np.array([9, 99])
        df2 = de.DataFrame({'a': [1, 5], 'b': ['eleni', 'teddy'], 'c': [nan, 5.4],
                            'd': [True, False], 'e': [9, 99]})
        assert_frame_equal(df1, df2)

        df1 = self.df.copy()
        df1[:, 'e'] = np.array([9, np.nan])
        df2 = de.DataFrame({'a': [1, 5], 'b': ['eleni', 'teddy'], 'c': [nan, 5.4],
                            'd': [True, False], 'e': [9, np.nan]})
        assert_frame_equal(df1, df2)

        df1 = self.df.copy()
        df1[:, 'e'] = np.array([True, False])
        df2 = de.DataFrame({'a': [1, 5], 'b': ['eleni', 'teddy'], 'c': [nan, 5.4],
                            'd': [True, False], 'e': [True, False]})
        assert_frame_equal(df1, df2)

        df1 = self.df.copy()
        df1[:, 'e'] = np.array(['poop', nan], dtype='O')
        df2 = de.DataFrame({'a': [1, 5], 'b': ['eleni', 'teddy'], 'c': [nan, 5.4],
                            'd': [True, False], 'e': ['poop', nan]})
        assert_frame_equal(df1, df2)

        df1 = self.df.copy()
        df1[:, 'e'] = np.array(['poop', 'pants'])
        df2 = de.DataFrame({'a': [1, 5], 'b': ['eleni', 'teddy'], 'c': [nan, 5.4],
                            'd': [True, False], 'e': ['poop', 'pants']})
        assert_frame_equal(df1, df2)

        df1 = self.df.copy()
        df1[:, 'e'] = np.array([nan, nan])
        df2 = de.DataFrame({'a': [1, 5], 'b': ['eleni', 'teddy'], 'c': [nan, 5.4],
                            'd': [True, False], 'e': [nan, nan]})
        assert_frame_equal(df1, df2)

    def test_setitem_entire_new_colunm_from_list(self):
        df1 = self.df.copy()
        df1[:, 'e'] = [9, 99]
        df2 = de.DataFrame({'a': [1, 5], 'b': ['eleni', 'teddy'], 'c': [nan, 5.4],
                            'd': [True, False], 'e': [9, 99]})
        assert_frame_equal(df1, df2)

        df1 = self.df.copy()
        df1[:, 'e'] = [9, np.nan]
        df2 = de.DataFrame({'a': [1, 5], 'b': ['eleni', 'teddy'], 'c': [nan, 5.4],
                            'd': [True, False], 'e': [9, np.nan]})
        assert_frame_equal(df1, df2)

        df1 = self.df.copy()
        df1[:, 'e'] = [True, False]
        df2 = de.DataFrame({'a': [1, 5], 'b': ['eleni', 'teddy'], 'c': [nan, 5.4],
                            'd': [True, False], 'e': [True, False]})
        assert_frame_equal(df1, df2)

        df1 = self.df.copy()
        df1[:, 'e'] = ['poop', nan]
        df2 = de.DataFrame({'a': [1, 5], 'b': ['eleni', 'teddy'], 'c': [nan, 5.4],
                            'd': [True, False], 'e': ['poop', nan]})
        assert_frame_equal(df1, df2)

        df1 = self.df.copy()
        df1[:, 'e'] = ['poop', 'pants']
        df2 = de.DataFrame({'a': [1, 5], 'b': ['eleni', 'teddy'], 'c': [nan, 5.4],
                            'd': [True, False], 'e': ['poop', 'pants']})
        assert_frame_equal(df1, df2)

        df1 = self.df.copy()
        df1[:, 'e'] = [nan, nan]
        df2 = de.DataFrame({'a': [1, 5], 'b': ['eleni', 'teddy'], 'c': [nan, 5.4],
                            'd': [True, False], 'e': [nan, nan]})
        assert_frame_equal(df1, df2)

    def test_setitem_entire_old_column_from_array(self):
        df1 = self.df.copy()
        df1[:, 'd'] = np.array([9, 99])
        df2 = de.DataFrame({'a': [1, 5], 'b': ['eleni', 'teddy'], 'c': [nan, 5.4],
                            'd': [9, 99]})
        assert_frame_equal(df1, df2)

        df1 = self.df.copy()
        df1[:, 'd'] = np.array([9, np.nan])
        df2 = de.DataFrame({'a': [1, 5], 'b': ['eleni', 'teddy'], 'c': [nan, 5.4],
                            'd': [9, np.nan]})
        assert_frame_equal(df1, df2)

        df1 = self.df.copy()
        df1[:, 'd'] = np.array([True, False])
        df2 = de.DataFrame({'a': [1, 5], 'b': ['eleni', 'teddy'], 'c': [nan, 5.4],
                            'd': [True, False]})
        assert_frame_equal(df1, df2)

        df1 = self.df.copy()
        df1[:, 'd'] = np.array(['poop', nan], dtype='O')
        df2 = de.DataFrame({'a': [1, 5], 'b': ['eleni', 'teddy'], 'c': [nan, 5.4],
                            'd': ['poop', nan]})
        assert_frame_equal(df1, df2)

        df1 = self.df.copy()
        df1[:, 'a'] = np.array(['poop', 'pants'], dtype='O')
        df2 = de.DataFrame({'a': ['poop', 'pants'], 'b': ['eleni', 'teddy'], 'c': [nan, 5.4],
                            'd': [True, False]})
        assert_frame_equal(df1, df2)

        df1 = self.df.copy()
        df1[:, 'b'] = np.array([nan, nan])
        df2 = de.DataFrame({'a': [1, 5], 'b': [nan, nan], 'c': [nan, 5.4],
                            'd': [True, False]})
        assert_frame_equal(df1, df2)

        df1 = self.df.copy()
        df1[:, 'c'] = np.array([False, False])
        df2 = de.DataFrame({'a': [1, 5], 'b': ['eleni', 'teddy'], 'c': [False, False],
                           'd': [True, False]})
        assert_frame_equal(df1, df2)

        with pytest.raises(ValueError):
            df1[:, 'b'] = np.array([1, 2, 3])

        with pytest.raises(ValueError):
            df1[:, 'b'] = np.array([1])

        with pytest.raises(TypeError):
            df1[:, 'a'] = np.array([5, {1, 2, 3}])

    def test_setitem_entire_old_column_from_list(self):
        df1 = self.df.copy()
        df1[:, 'd'] = [9, 99]
        df2 = de.DataFrame({'a': [1, 5], 'b': ['eleni', 'teddy'], 'c': [nan, 5.4],
                            'd': [9, 99]})
        assert_frame_equal(df1, df2)

        df1 = self.df.copy()
        df1[:, 'd'] = [9, np.nan]
        df2 = de.DataFrame({'a': [1, 5], 'b': ['eleni', 'teddy'], 'c': [nan, 5.4],
                            'd': [9, np.nan]})
        assert_frame_equal(df1, df2)

        df1 = self.df.copy()
        df1[:, 'd'] = [True, False]
        df2 = de.DataFrame({'a': [1, 5], 'b': ['eleni', 'teddy'], 'c': [nan, 5.4],
                            'd': [True, False]})
        assert_frame_equal(df1, df2)

        df1 = self.df.copy()
        df1[:, 'd'] = ['poop', nan]
        df2 = de.DataFrame({'a': [1, 5], 'b': ['eleni', 'teddy'], 'c': [nan, 5.4],
                            'd': ['poop', nan]})
        assert_frame_equal(df1, df2)

        df1 = self.df.copy()
        df1[:, 'a'] = ['poop', 'pants']
        df2 = de.DataFrame({'a': ['poop', 'pants'], 'b': ['eleni', 'teddy'], 'c': [nan, 5.4],
                            'd': [True, False]})
        assert_frame_equal(df1, df2)

        df1 = self.df.copy()
        df1[:, 'b'] = [nan, nan]
        df2 = de.DataFrame({'a': [1, 5], 'b': [nan, nan], 'c': [nan, 5.4],
                            'd': [True, False]})
        assert_frame_equal(df1, df2)

        df1 = self.df.copy()
        df1[:, 'c'] = [False, False]
        df2 = de.DataFrame({'a': [1, 5], 'b': ['eleni', 'teddy'], 'c': [False, False],
                           'd': [True, False]})
        assert_frame_equal(df1, df2)

        with pytest.raises(ValueError):
            self.df[:, 'b'] = [1, 2, 3]

        with pytest.raises(ValueError):
            self.df[:, 'b'] = [1]

        with pytest.raises(TypeError):
            self.df[:, 'a'] = [5, {1, 2, 3}]

    def test_setitem_simultaneous_row_and_column(self):
        df1 = self.df1.copy()
        df1[[0, 1], 'a'] = [9, 10]
        df2 = de.DataFrame({'a': [9, 10, 7, 11], 'b': ['eleni', 'teddy', 'niko', 'penny'],
                            'c': [nan, 5.4, -1.1, .045], 'd': [True, False, False, True]})
        assert_frame_equal(df1, df2)

        df1 = self.df1.copy()
        df1[[0, -1], 'a'] = np.array([9, 10.5])
        df2 = de.DataFrame({'a': [9, 5, 7, 10.5], 'b': ['eleni', 'teddy', 'niko', 'penny'],
                            'c': [nan, 5.4, -1.1, .045], 'd': [True, False, False, True]})
        assert_frame_equal(df1, df2)

        df1 = self.df1.copy()
        df1[2:, 'b'] = np.array(['NIKO', 'PENNY'])
        df2 = de.DataFrame({'a': [1, 5, 7, 11], 'b': ['eleni', 'teddy', 'NIKO', 'PENNY'],
                            'c': [nan, 5.4, -1.1, .045], 'd': [True, False, False, True]})
        assert_frame_equal(df1, df2)

        df1 = self.df1.copy()
        df1[2, ['b', 'c']] = ['NIKO', 9.3]
        df2 = de.DataFrame({'a': [1, 5, 7, 11], 'b': ['eleni', 'teddy', 'NIKO', 'penny'],
                            'c': [nan, 5.4, 9.3, .045], 'd': [True, False, False, True]})
        assert_frame_equal(df1, df2)

        df1 = self.df1.copy()
        df1[2, ['c', 'b']] = [9.3, nan]

        df2 = de.DataFrame({'a': [1, 5, 7, 11], 'b': ['eleni', 'teddy', nan, 'penny'],
                            'c': [nan, 5.4, 9.3, .045], 'd': [True, False, False, True]})
        assert_frame_equal(df1, df2)

        df1 = self.df1.copy()
        df1[[1, -1], 'b':'d'] = [['TEDDY', nan, True], [nan, 5.5, False]]

        df2 = de.DataFrame({'a': [1, 5, 7, 11], 'b': ['eleni', 'TEDDY', 'niko', nan],
                            'c': [nan, nan, -1.1, 5.5], 'd': [True, True, False, False]})
        assert_frame_equal(df1, df2)

        df1 = self.df1.copy()
        df1[1:-1, 'a':'d':2] = [[nan, 4], [3, 99]]

        df2 = de.DataFrame({'a': [1, nan, 3, 11], 'b': ['eleni', 'teddy', 'niko', 'penny'],
                            'c': [nan, 4, 99, .045], 'd': [True, False, False, True]})
        assert_frame_equal(df1, df2)

    def test_testitem_boolean(self):
        df1 = self.df1.copy()
        criteria = df1[:, 'a'] > 4
        df1[criteria, 'b'] = 'TEDDY'
        df2 = de.DataFrame({'a': [1, 5, 7, 11], 'b': ['eleni', 'TEDDY', 'TEDDY', 'TEDDY'],
                            'c': [nan, 5.4, -1.1, .045], 'd': [True, False, False, True]})
        assert_frame_equal(df1, df2)

        df1 = self.df1.copy()
        criteria = df1[:, 'a'] > 4
        df1[criteria, 'b'] = ['A', 'B', 'C']
        df2 = de.DataFrame({'a': [1, 5, 7, 11], 'b': ['eleni', 'A', 'B', 'C'],
                            'c': [nan, 5.4, -1.1, .045], 'd': [True, False, False, True]})
        assert_frame_equal(df1, df2)


        df1 = de.DataFrame({'a': [1, 5, 7, 11], 'b': ['eleni', 'teddy', 'niko', 'penny'],
                            'c': [nan, 5.4, -1.1, .045], 'd': [True, False, False, True]})


