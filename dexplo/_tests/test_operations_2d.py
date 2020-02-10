import dexplo as dx
import numpy as np
from numpy import array, nan
import pytest
from dexplo.testing import assert_frame_equal


class TestArithmeticOperatorsDF:
    a = [1, 2]
    b = [-10, 10]
    c = [1.5, 8]
    d = [2.3, np.nan]
    e = list('ab')
    f = [True, False]
    g = [np.timedelta64(x, 'D') for x in range(2)]
    df = dx.DataFrame({'a': a,
                       'b': b,
                       'c': c,
                       'd': d,
                       'e': e,
                       'f': f,
                       'g': g},
                      columns=list('abcdefg'))

    a = [5]
    b = [99]
    c = [2.1]
    d = [np.nan]
    e = ['twoplustwo']
    f = [True]
    g = [np.timedelta64(1000, 'D')]
    df_one_row = dx.DataFrame({'a': a,
                               'b': b,
                               'c': c,
                               'd': d,
                               'e': e,
                               'f': f,
                               'g': g},
                              columns=list('abcdefg'))

    df_one_row_number = df_one_row.select_dtypes('number')
    df_one_col = dx.DataFrame({'COL': [5, 2.1]})
    df_number = df.select_dtypes('number')

    df_number2 = dx.DataFrame({'A': [4, 5],
                               'B': [0, 0],
                               'C': [2, 2],
                               'D': [-2, 4]},
                              columns=list('ABCD'))

    df_strings = dx.DataFrame({'a': ['one', 'two'], 'b': ['three', 'four']})
    df_strings_row = dx.DataFrame({'a': ['MOOP'], 'b': ['DOOP']})
    df_strings_col = dx.DataFrame({'a': ['MOOP', 'DOOP']})

    def test_add_df(self):
        df_answer = dx.DataFrame({'a': np.array([2, 4]),
                                  'b': np.array([-20, 20]),
                                  'c': np.array([3., 16.]),
                                  'd': np.array([4.6, nan]),
                                  'e': np.array(['aa', 'bb'], dtype=object),
                                  'f': np.array([True, False]),
                                  'g': np.array([0, 172800000000000], dtype='timedelta64[ns]')})
        assert_frame_equal(self.df + self.df, df_answer)

        df_answer = dx.DataFrame({'a': array([5, 7]),
                                  'b': array([-10, 10]),
                                  'c': array([3.5, 10.]),
                                  'd': array([0.3, nan])})
        df_result = self.df_number + self.df_number2
        assert_frame_equal(df_result, df_answer)

    def test_add_one_col(self):
        df_answer = dx.DataFrame({'a': np.array([6., 4.1]),
                                  'b': np.array([-5., 12.1]),
                                  'c': np.array([6.5, 10.1]),
                                  'd': np.array([7.3, nan])})
        df_result = self.df_number + self.df_one_col
        assert_frame_equal(df_result, df_answer)

        df_result = self.df_one_col + self.df_number
        assert_frame_equal(df_result, df_answer)

    def test_add_one_row(self):
        df_answer = dx.DataFrame({'a': array([6, 7]),
                                  'b': array([ 89, 109]),
                                  'c': array([ 3.6, 10.1]),
                                  'd': array([nan, nan])})
        df_result = self.df_number + self.df_one_row_number
        assert_frame_equal(df_result, df_answer)

        df_result = self.df_number + self.df_one_row_number
        assert_frame_equal(df_answer, df_result)

    def test_add_string(self):
        df_answer = dx.DataFrame({'a': array(['oneone', 'twotwo'], dtype=object),
                                  'b': array(['threethree', 'fourfour'], dtype=object)})
        df_result = self.df_strings + self.df_strings
        assert_frame_equal(df_answer, df_result)

    def test_add_string_row(self):
        df_answer = dx.DataFrame({'a': array(['oneMOOP', 'twoMOOP'], dtype=object),
                                  'b': array(['threeDOOP', 'fourDOOP'], dtype=object)})
        df_result = self.df_strings + self.df_strings_row
        assert_frame_equal(df_answer, df_result)

        df_answer = dx.DataFrame({'a': array(['MOOPone', 'MOOPtwo'], dtype=object),
                                  'b': array(['DOOPthree', 'DOOPfour'], dtype=object)})
        df_result = self.df_strings_row + self.df_strings
        assert_frame_equal(df_answer, df_result)


class TestMultipleBooleanConditions:
    df = dx.DataFrame({'a': [1, 4, 10, 20],
                       'b': ['a', 'a', 'c', 'c'],
                       'c': [5, 1, 14, 3]})

    def test_and(self):
        df1 = (self.df[:, 'a'] > 5) & (self.df[:, 'a'] < 15)
        df2 = dx.DataFrame({'a': [False, False, True, False]})
        assert_frame_equal(df1, df2)

    def test_or(self):
        df1 = (self.df[:, 'a'] > 5) | (self.df[:, 'c'] < 2)
        df2 = dx.DataFrame({'a': [False, True, True, True]})
        assert_frame_equal(df1, df2)

    def test_invert(self):
        df1 = ~((self.df[:, 'a'] > 5) | (self.df[:, 'c'] < 2))
        df2 = dx.DataFrame({'a': [True, False, False, False]})
        assert_frame_equal(df1, df2)
