import dexplo as de
import numpy as np
from numpy import nan, array
import pytest
from dexplo.testing import assert_frame_equal, assert_array_equal


class TestSortValues(object):

    def test_sort_values_one(self):
        data = {'a': [4, 3, nan, 6, 3, 2],
                'b': [None, 'f', 'd', 'f', 'd', 'er'],
                'c': [12, 444, -5.6, 5, 1, 7]}
        df = de.DataFrame(data)

        df1 = df.sort_values('a')
        df2 = de.DataFrame(data={'a': [2.0, 3.0, 3.0, 4.0, 6.0, nan],
                                 'b': ['er', 'f', 'd', None, 'f', 'd'],
                                 'c': [7.0, 444.0, 1.0, 12.0, 5.0, -5.6]})
        assert_frame_equal(df1, df2)

        df1 = df.sort_values('b')
        df2 = de.DataFrame({'a': [nan, 3.0, 2.0, 3.0, 6.0, 4.0],
                            'b': ['d', 'd', 'er', 'f', 'f', None],
                            'c': [-5.6, 1.0, 7.0, 444.0, 5.0, 12.0]})
        assert_frame_equal(df1, df2)

        df = de.DataFrame({'a': [2, 3, nan, 6, 3, 2],
                           'b': [None, 'f', 'd', 'f', 'd', 'er'],
                           'c': [12, 444, -5.6, 5, 1, 7]})

        df1 = df.sort_values('b', ascending=False)
        df2 = de.DataFrame({'a': [3.0, 6.0, 2.0, nan, 3.0, 2.0],
                            'b': ['f', 'f', 'er', 'd', 'd', None],
                            'c': [444.0, 5.0, 7.0, -5.6, 1.0, 12.0]})
        assert_frame_equal(df1, df2)

        df1 = df.sort_values('a', ascending=False)
        df2 = de.DataFrame({'a': [6.0, 3.0, 3.0, 2.0, 2.0, nan],
                            'b': ['f', 'f', 'd', None, 'er', 'd'],
                            'c': [5.0, 444.0, 1.0, 12.0, 7.0, -5.6]})
        assert_frame_equal(df1, df2)

    def test_sort_values_multiple(self):
        df = de.DataFrame({'a': [2, 3, nan, 6, 3, 2],
                           'b': [None, 'f', 'd', 'f', 'd', 'er'],
                           'c': [12, 444, -5.6, 5, 1, 7]})

        df1 = df.sort_values(['a', 'b'], ascending=False)
        df2 = de.DataFrame({'a': [6.0, 3.0, 3.0, 2.0, 2.0, nan],
                            'b': ['f', 'f', 'd', 'er', None, 'd'],
                            'c': [5.0, 444.0, 1.0, 7.0, 12.0, -5.6]})
        assert_frame_equal(df1, df2)

        df1 = df.sort_values(['a', 'b'], ascending=True)
        df2 = de.DataFrame({'a': [2.0, 2.0, 3.0, 3.0, 6.0, nan],
                            'b': ['er', None, 'd', 'f', 'f', 'd'],
                            'c': [7.0, 12.0, 1.0, 444.0, 5.0, -5.6]})
        assert_frame_equal(df1, df2)

        df1 = df.sort_values(['a', 'b'], ascending=[True, False])
        df2 = de.DataFrame({'a': [2.0, 2.0, 3.0, 3.0, 6.0, nan],
                            'b': ['er', None, 'f', 'd', 'f', 'd'],
                            'c': [7.0, 12.0, 444.0, 1.0, 5.0, -5.6]})
        assert_frame_equal(df1, df2)

        df1 = df.sort_values(['a', 'b'], ascending=[False, True])
        df2 = de.DataFrame({'a': [6.0, 3.0, 3.0, 2.0, 2.0, nan],
                            'b': ['f', 'd', 'f', 'er', None, 'd'],
                            'c': [5.0, 1.0, 444.0, 7.0, 12.0, -5.6]})
        assert_frame_equal(df1, df2)

        df1 = df.sort_values(['b', 'a'], ascending=[False, True])
        df2 = de.DataFrame({'a': [3.0, 6.0, 2.0, 3.0, nan, 2.0],
                            'b': ['f', 'f', 'er', 'd', 'd', None],
                            'c': [444.0, 5.0, 7.0, 1.0, -5.6, 12.0]})
        assert_frame_equal(df1, df2)


class TestRank:

    def test_rank_min(self):
        df = de.DataFrame({'a': [2, 3, nan, 6, 3, 2],
                           'b': [None, 'f', 'd', 'f', 'd', 'er'],
                           'c': [12, 444, -5.6, 5, 1, 7]})
        df1 = df.rank()
        df2 = de.DataFrame({'a': [1.0, 3.0, nan, 5.0, 3.0, 1.0],
                            'b': [nan, 4.0, 1.0, 4.0, 1.0, 3.0],
                            'c': [5.0, 6.0, 1.0, 3.0, 2.0, 4.0]})
        assert_frame_equal(df1, df2)

        df1 = df.rank(na_option='top')
        df2 = de.DataFrame({'a': [2, 4, 1, 6, 4, 2], 'b': [1, 5, 2, 5, 2, 4],
                            'c': [5, 6, 1, 3, 2, 4]})
        assert_frame_equal(df1, df2)

        df1 = df.rank(na_option='bottom')
        df2 = de.DataFrame({'a': [1, 3, 6, 5, 3, 1], 'b': [6, 4, 1, 4, 1, 3],
                            'c': [5, 6, 1, 3, 2, 4]})
        assert_frame_equal(df1, df2)

        df1 = df.rank(na_option='keep', ascending=False)
        df2 = de.DataFrame({'a': [4.0, 2.0, nan, 1.0, 2.0, 4.0],
                            'b': [nan, 1.0, 4.0, 1.0, 4.0, 3.0],
                            'c': [2.0, 1.0, 6.0, 4.0, 5.0, 3.0]})
        assert_frame_equal(df1, df2)

        df1 = df.rank(na_option='top', ascending=False)
        df2 = de.DataFrame({'a': [5, 3, 1, 2, 3, 5], 'b': [1, 2, 5, 2, 5, 4],
                            'c': [2, 1, 6, 4, 5, 3]})
        assert_frame_equal(df1, df2)

        df1 = df.rank(na_option='bottom', ascending=False)
        df2 = de.DataFrame(
            {'a': [4, 2, 6, 1, 2, 4], 'b': [6, 1, 4, 1, 4, 3], 'c': [2, 1, 6, 4, 5, 3]})
        assert_frame_equal(df1, df2)

    def test_rank_max(self):
        df = de.DataFrame({'a': [2, 3, nan, 6, 3, 2],
                           'b': [None, 'f', 'd', 'f', 'd', 'er'],
                           'c': [12, 444, -5.6, 5, 1, 7]})

        df1 = df.rank(method='max', na_option='keep', ascending=True)
        df2 = de.DataFrame({'a': [2.0, 4.0, nan, 5.0, 4.0, 2.0],
                            'b': [nan, 5.0, 2.0, 5.0, 2.0, 3.0],
                            'c': [5.0, 6.0, 1.0, 3.0, 2.0, 4.0]})
        assert_frame_equal(df1, df2)

        df1 = df.rank(method='max', na_option='keep', ascending=False)
        df2 = de.DataFrame({'a': [5.0, 3.0, nan, 1.0, 3.0, 5.0],
                            'b': [nan, 2.0, 5.0, 2.0, 5.0, 3.0],
                            'c': [2.0, 1.0, 6.0, 4.0, 5.0, 3.0]})
        assert_frame_equal(df1, df2)

        df1 = df.rank(method='max', na_option='top', ascending=True)
        df2 = de.DataFrame({'a': [3, 5, 1, 6, 5, 3], 'b': [1, 6, 3, 6, 3, 4],
                            'c': [5, 6, 1, 3, 2, 4]})
        assert_frame_equal(df1, df2)

        df1 = df.rank(method='max', na_option='top', ascending=False)
        df2 = de.DataFrame({'a': [6, 4, 1, 2, 4, 6], 'b': [1, 3, 6, 3, 6, 4],
                            'c': [2, 1, 6, 4, 5, 3]})
        assert_frame_equal(df1, df2)

        df1 = df.rank(method='max', na_option='bottom', ascending=True)
        df2 = de.DataFrame({'a': [2, 4, 6, 5, 4, 2], 'b': [6, 5, 2, 5, 2, 3],
                            'c': [5, 6, 1, 3, 2, 4]})
        assert_frame_equal(df1, df2)

        df1 = df.rank(method='max', na_option='bottom', ascending=False)
        df2 = de.DataFrame({'a': [5, 3, 6, 1, 3, 5], 'b': [6, 2, 5, 2, 5, 3],
                            'c': [2, 1, 6, 4, 5, 3]})
        assert_frame_equal(df1, df2)

    def test_rank_dense(self):
        df = de.DataFrame({'a': [2, 3, nan, 6, 3, 2],
                           'b': [None, 'f', 'd', 'f', 'd', 'er'],
                           'c': [12, 444, -5.6, 5, 1, 7]})

        df1 = df.rank(method='dense', na_option='keep', ascending=True)
        df2 = de.DataFrame({'a': [1.0, 2.0, nan, 3.0, 2.0, 1.0],
                            'b': [nan, 3.0, 1.0, 3.0, 1.0, 2.0],
                            'c': [5.0, 6.0, 1.0, 3.0, 2.0, 4.0]})
        assert_frame_equal(df1, df2)

        df1 = df.rank(method='dense', na_option='keep', ascending=False)
        df2 = de.DataFrame({'a': [3.0, 2.0, nan, 1.0, 2.0, 3.0],
                            'b': [nan, 1.0, 3.0, 1.0, 3.0, 2.0],
                            'c': [2.0, 1.0, 6.0, 4.0, 5.0, 3.0]})
        assert_frame_equal(df1, df2)

        df1 = df.rank(method='dense', na_option='top', ascending=True)
        df2 = de.DataFrame({'a': [2, 3, 1, 4, 3, 2], 'b': [1, 4, 2, 4, 2, 3],
                            'c': [5, 6, 1, 3, 2, 4]})
        assert_frame_equal(df1, df2)

        df1 = df.rank(method='dense', na_option='top', ascending=False)
        df2 = de.DataFrame(
            {'a': [4, 3, 1, 2, 3, 4], 'b': [1, 2, 4, 2, 4, 3], 'c': [2, 1, 6, 4, 5, 3]})
        assert_frame_equal(df1, df2)

        df1 = df.rank(method='dense', na_option='bottom', ascending=True)
        df2 = de.DataFrame(
            {'a': [1, 2, 4, 3, 2, 1], 'b': [4, 3, 1, 3, 1, 2], 'c': [5, 6, 1, 3, 2, 4]})
        assert_frame_equal(df1, df2)

        df1 = df.rank(method='dense', na_option='bottom', ascending=False)
        df2 = de.DataFrame(
            {'a': [3, 2, 4, 1, 2, 3], 'b': [4, 1, 3, 1, 3, 2], 'c': [2, 1, 6, 4, 5, 3]})
        assert_frame_equal(df1, df2)

    def test_rank_first(self):
        df = de.DataFrame({'a': [2, 3, nan, 6, 3, 2],
                           'b': [None, 'f', 'd', 'f', 'd', 'er'],
                           'c': [12, 444, -5.6, 5, 1, 7]})

        df1 = df.rank(method='first', na_option='keep', ascending=True)
        df2 = de.DataFrame({'a': [1.0, 3.0, nan, 5.0, 4.0, 2.0],
                            'b': [nan, 4.0, 1.0, 5.0, 2.0, 3.0],
                            'c': [5.0, 6.0, 1.0, 3.0, 2.0, 4.0]})
        assert_frame_equal(df1, df2)

        df1 = df.rank(method='first', na_option='keep', ascending=False)
        df2 = de.DataFrame({'a': [4.0, 2.0, nan, 1.0, 3.0, 5.0],
                            'b': [nan, 1.0, 4.0, 2.0, 5.0, 3.0],
                            'c': [2.0, 1.0, 6.0, 4.0, 5.0, 3.0]})
        assert_frame_equal(df1, df2)

        df1 = df.rank(method='first', na_option='top', ascending=True)
        df2 = de.DataFrame(
            {'a': [2, 4, 1, 6, 5, 3], 'b': [1, 5, 2, 6, 3, 4], 'c': [5, 6, 1, 3, 2, 4]})
        assert_frame_equal(df1, df2)

        df1 = df.rank(method='first', na_option='top', ascending=False)
        df2 = de.DataFrame(
            {'a': [5, 3, 1, 2, 4, 6], 'b': [1, 2, 5, 3, 6, 4], 'c': [2, 1, 6, 4, 5, 3]})
        assert_frame_equal(df1, df2)

        df1 = df.rank(method='first', na_option='bottom', ascending=True)
        df2 = de.DataFrame(
            {'a': [1, 3, 6, 5, 4, 2], 'b': [6, 4, 1, 5, 2, 3], 'c': [5, 6, 1, 3, 2, 4]})
        assert_frame_equal(df1, df2)

        df1 = df.rank(method='first', na_option='bottom', ascending=False)
        df2 = de.DataFrame(
            {'a': [4, 2, 6, 1, 3, 5], 'b': [6, 1, 4, 2, 5, 3], 'c': [2, 1, 6, 4, 5, 3]})
        assert_frame_equal(df1, df2)

    def test_rank_average(self):
        df = de.DataFrame({'a': [2, 3, nan, 6, 3, 2],
                           'b': [None, 'f', 'd', 'f', 'd', 'er'],
                           'c': [12, 444, -5.6, 5, 1, 7]})

        df1 = df.rank(method='average', na_option='keep', ascending=True)
        df2 = de.DataFrame({'a': [1.5, 3.5, nan, 5.0, 3.5, 1.5],
                            'b': [nan, 4.5, 1.5, 4.5, 1.5, 3.0],
                            'c': [5.0, 6.0, 1.0, 3.0, 2.0, 4.0]})
        assert_frame_equal(df1, df2)

        df1 = df.rank(method='average', na_option='keep', ascending=False)
        df2 = de.DataFrame({'a': [4.5, 2.5, nan, 1.0, 2.5, 4.5],
                            'b': [nan, 1.5, 4.5, 1.5, 4.5, 3.0],
                            'c': [2.0, 1.0, 6.0, 4.0, 5.0, 3.0]})
        assert_frame_equal(df1, df2)

        df1 = df.rank(method='average', na_option='top', ascending=True)
        df2 = de.DataFrame({'a': [2.5, 4.5, 1.0, 6.0, 4.5, 2.5],
                            'b': [1.0, 5.5, 2.5, 5.5, 2.5, 4.0],
                            'c': [5.0, 6.0, 1.0, 3.0, 2.0, 4.0]})
        assert_frame_equal(df1, df2)

        df1 = df.rank(method='average', na_option='top', ascending=False)
        df2 = de.DataFrame({'a': [5.5, 3.5, 1.0, 2.0, 3.5, 5.5],
                            'b': [1.0, 2.5, 5.5, 2.5, 5.5, 4.0],
                            'c': [2.0, 1.0, 6.0, 4.0, 5.0, 3.0]})
        assert_frame_equal(df1, df2)

        df1 = df.rank(method='average', na_option='bottom', ascending=True)
        df2 = de.DataFrame({'a': [1.5, 3.5, 6.0, 5.0, 3.5, 1.5],
                            'b': [6.0, 4.5, 1.5, 4.5, 1.5, 3.0],
                            'c': [5.0, 6.0, 1.0, 3.0, 2.0, 4.0]})
        assert_frame_equal(df1, df2)

        df1 = df.rank(method='average', na_option='bottom', ascending=False)
        df2 = de.DataFrame({'a': [4.5, 2.5, 6.0, 1.0, 2.5, 4.5],
                            'b': [6.0, 1.5, 4.5, 1.5, 4.5, 3.0],
                            'c': [2.0, 1.0, 6.0, 4.0, 5.0, 3.0]})
        assert_frame_equal(df1, df2)


class TestStreak:

    def test_streak(self):
        df = de.DataFrame(
            {'AIRLINE': ['AA', 'AA', 'AA', 'UA', 'DL', 'DL', 'WN', 'WN', 'WN', 'AS', None],
             'DAY_OF_WEEK': [2, 3, 6, 6, 6, 6, 4, 4, 1, 2, 2],
             'DEPARTURE_DELAY': [nan, nan, -1.0, -1.0, -1.0, 22.0, 3.0, 3.0, 21.0,
                                 -2.0, nan]})
        arr1 = df.streak('AIRLINE')
        arr2 = array([1, 2, 3, 1, 1, 2, 1, 2, 3, 1, 1])
        assert_array_equal(arr1, arr2)

        arr1 = df.streak('DAY_OF_WEEK')
        arr2 = array([1, 1, 1, 2, 3, 4, 1, 2, 1, 1, 2])
        assert_array_equal(arr1, arr2)

        arr1 = df.streak('DEPARTURE_DELAY')
        arr2 = array([1, 1, 1, 2, 3, 1, 1, 2, 1, 1, 1])
        assert_array_equal(arr1, arr2)

    def test_streak_value(self):
        df = de.DataFrame(
            {'AIRLINE': ['AA', 'AA', 'AA', 'UA', 'DL', 'DL', 'WN', 'WN', 'WN', 'AS', None],
             'DAY_OF_WEEK': [2, 3, 6, 6, 6, 6, 4, 4, 1, 2, 2],
             'DEPARTURE_DELAY': [nan, nan, -1.0, -1.0, -1.0, 22.0, 3.0, 3.0, 21.0,
                                 -2.0, nan]})

        with pytest.raises(TypeError):
            df.streak('DEPARTURE_DELAY', 'AA')

        arr1 = df.streak('DEPARTURE_DELAY', -1)
        arr2 = array([0, 0, 1, 2, 3, 0, 0, 0, 0, 0, 0])
        assert_array_equal(arr1, arr2)

        arr1 = df.streak('DAY_OF_WEEK', 6)
        arr2 = array([0, 0, 1, 2, 3, 4, 0, 0, 0, 0, 0])
        assert_array_equal(arr1, arr2)

    def test_streak_group(self):
        df = de.DataFrame(
            {'AIRLINE': ['AA', 'AA', 'AA', 'UA', 'DL', 'DL', 'WN', 'WN', 'AA', 'AA', None],
             'DAY_OF_WEEK': [2, 3, 6, 6, 6, 6, 4, 4, 1, 6, 6],
             'DEPARTURE_DELAY': [nan, nan, -1.0, -1.0, -1.0, 22.0, 3.0, 3.0, 21.0,
                                 -2.0, nan]})
        arr1 = df.streak('AIRLINE', group=True)
        arr2 = array([1, 1, 1, 2, 3, 3, 4, 4, 5, 5, 6])
        assert_array_equal(arr1, arr2)

        arr1 = df.streak('DEPARTURE_DELAY', group=True)
        arr2 = array([1, 2, 3, 3, 3, 4, 5, 5, 6, 7, 8])
        assert_array_equal(arr1, arr2)


class TestDrop:

    def test_drop(self):
        data = {'a': [0, 0, 5, 9],
                'b': [0, 1.5, 8, 9],
                'c': [''] + list('efs'),
                'd': [False, False, True, False],
                'e': [0, 20, 30, 4],
                'f': ['a', nan, 'ad', 'effd'],
                'g': [np.nan] * 4}
        df = de.DataFrame(data)

        df1 = df.drop(columns='b')
        df2 = de.DataFrame({'a': [0, 0, 5, 9],
                            'c': ['', 'e', 'f', 's'],
                            'd': [False, False, True, False],
                            'e': [0, 20, 30, 4],
                            'f': ['a', None, 'ad', 'effd'],
                            'g': [nan, nan, nan, nan]})
        assert_frame_equal(df1, df2)

        df1 = df.drop(columns='g')
        df2 = de.DataFrame({'a': [0, 0, 5, 9],
                            'b': [0.0, 1.5, 8.0, 9.0],
                            'c': ['', 'e', 'f', 's'],
                            'd': [False, False, True, False],
                            'e': [0, 20, 30, 4],
                            'f': ['a', None, 'ad', 'effd']})
        assert_frame_equal(df1, df2)

        df1 = df.drop(columns='d')
        df2 = de.DataFrame({'a': [0, 0, 5, 9],
                            'b': [0.0, 1.5, 8.0, 9.0],
                            'c': ['', 'e', 'f', 's'],
                            'e': [0, 20, 30, 4],
                            'f': ['a', None, 'ad', 'effd'],
                            'g': [nan, nan, nan, nan]})
        assert_frame_equal(df1, df2)

        df1 = df.drop(columns=list('abcdefg'))
        df2 = de.DataFrame({})
        assert_frame_equal(df1, df2)

        df1 = df.drop(columns=list('ade'))
        df2 = de.DataFrame({'b': [0.0, 1.5, 8.0, 9.0],
                            'c': ['', 'e', 'f', 's'],
                            'f': ['a', None, 'ad', 'effd'],
                            'g': [nan, nan, nan, nan]})
        assert_frame_equal(df1, df2)

    def test_drop_rows(self):
        data = {'a': [0, 0, 5, 9],
                'b': [0, 1.5, 8, 9],
                'c': [''] + list('efs'),
                'd': [False, False, True, False],
                'e': [0, 20, 30, 4],
                'f': ['a', nan, 'ad', 'effd'],
                'g': [np.nan] * 4}
        df = de.DataFrame(data)

        df1 = df.drop(rows=3)
        df2 = de.DataFrame({'a': [0, 0, 5],
                            'b': [0.0, 1.5, 8.0],
                            'c': ['', 'e', 'f'],
                            'd': [False, False, True],
                            'e': [0, 20, 30],
                            'f': ['a', None, 'ad'],
                            'g': [nan, nan, nan]})
        assert_frame_equal(df1, df2)

        with pytest.raises(IndexError):
            df.drop(rows=5)

        df1 = df.drop(rows=[-1, 0, -3])
        df2 = de.DataFrame({'a': [5],
                            'b': [8.0],
                            'c': ['f'],
                            'd': [True],
                            'e': [30],
                            'f': ['ad'],
                            'g': [nan]})
        assert_frame_equal(df1, df2)

    def test_drop_rows_and_cols(self):
        data = {'a': [0, 0, 5, 9],
                'b': [0, 1.5, 8, 9],
                'c': [''] + list('efs'),
                'd': [False, False, True, False],
                'e': [0, 20, 30, 4],
                'f': ['a', nan, 'ad', 'effd'],
                'g': [np.nan] * 4}
        df = de.DataFrame(data)

        df1 = df.drop(1, 1)
        df2 = de.DataFrame({'a': [0, 5, 9],
                            'c': ['', 'f', 's'],
                            'd': [False, True, False],
                            'e': [0, 30, 4],
                            'f': ['a', 'ad', 'effd'],
                            'g': [nan, nan, nan]})
        assert_frame_equal(df1, df2)

        df1 = df.drop(-2, list('abc'))
        df2 = de.DataFrame({'d': [False, False, False],
                            'e': [0, 20, 4],
                            'f': ['a', None, 'effd'],
                            'g': [nan, nan, nan]})
        assert_frame_equal(df1, df2)

        df1 = df.drop([0, 3], [3, 'a', -2])
        df2 = de.DataFrame({'b': [1.5, 8.0], 'c': ['e', 'f'], 'e': [20, 30], 'g': [nan, nan]})
        assert_frame_equal(df1, df2)


class TestRename:

    def test_rename(self):
        data = {'a': [0, 0, 5, 9],
                'b': [0, 1.5, 8, 9],
                'c': [''] + list('efs'),
                'd': [False, False, True, False],
                'e': [0, 20, 30, 4],
                'f': ['a', nan, 'ad', 'effd'],
                'g': [np.nan] * 4}
        df = de.DataFrame(data)

        with pytest.raises(ValueError):
            df.rename({'a': 'b'})

        df1 = df.rename({'a': 'alpha'})
        df2 = de.DataFrame({'alpha': [0, 0, 5, 9],
                            'b': [0.0, 1.5, 8.0, 9.0],
                            'c': ['', 'e', 'f', 's'],
                            'd': [False, False, True, False],
                            'e': [0, 20, 30, 4],
                            'f': ['a', None, 'ad', 'effd'],
                            'g': [nan, nan, nan, nan]})
        assert_frame_equal(df1, df2)

        with pytest.raises(TypeError):
            df.rename({'a': 5})

        df1 = df.rename({'a': 'b', 'b': 't'})
        df2 = de.DataFrame({'b': [0, 0, 5, 9],
                            't': [0.0, 1.5, 8.0, 9.0],
                            'c': ['', 'e', 'f', 's'],
                            'd': [False, False, True, False],
                            'e': [0, 20, 30, 4],
                            'f': ['a', None, 'ad', 'effd'],
                            'g': [nan, nan, nan, nan]})
        assert_frame_equal(df1, df2)

        df1 = df.rename(list('poiuytr'))
        df2 = de.DataFrame({'p': [0, 0, 5, 9], 'o': [0.0, 1.5, 8.0, 9.0], 'i': ['', 'e', 'f', 's'],
                            'u': [False, False, True, False], 'y': [0, 20, 30, 4],
                            't': ['a', None, 'ad', 'effd'], 'r': [nan, nan, nan, nan]})
        assert_frame_equal(df1, df2)

class TestNLargest:

    def test_nlargest(self):
