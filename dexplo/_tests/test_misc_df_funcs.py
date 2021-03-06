import dexplo as dx
import numpy as np
from numpy import nan, array
import pytest
from dexplo.testing import assert_frame_equal, assert_array_equal


class TestSortValues(object):

    def test_sort_values_one(self):
        data = {'a': [4, 3, nan, 6, 3, 2],
                'b': [None, 'f', 'd', 'f', 'd', 'er'],
                'c': [12, 444, -5.6, 5, 1, 7]}
        df = dx.DataFrame(data)

        df1 = df.sort_values('a')
        df2 = dx.DataFrame(data={'a': [2.0, 3.0, 3.0, 4.0, 6.0, nan],
                                 'b': ['er', 'f', 'd', None, 'f', 'd'],
                                 'c': [7.0, 444.0, 1.0, 12.0, 5.0, -5.6]})
        assert_frame_equal(df1, df2)

        df1 = df.sort_values('b')
        df2 = dx.DataFrame({'a': [nan, 3.0, 2.0, 3.0, 6.0, 4.0],
                            'b': ['d', 'd', 'er', 'f', 'f', None],
                            'c': [-5.6, 1.0, 7.0, 444.0, 5.0, 12.0]})
        assert_frame_equal(df1, df2)

        df = dx.DataFrame({'a': [2, 3, nan, 6, 3, 2],
                           'b': [None, 'f', 'd', 'f', 'd', 'er'],
                           'c': [12, 444, -5.6, 5, 1, 7]})

        df1 = df.sort_values('b', ascending=False)
        df2 = dx.DataFrame({'a': [3.0, 6.0, 2.0, nan, 3.0, 2.0],
                            'b': ['f', 'f', 'er', 'd', 'd', None],
                            'c': [444.0, 5.0, 7.0, -5.6, 1.0, 12.0]})
        assert_frame_equal(df1, df2)

        df1 = df.sort_values('a', ascending=False)
        df2 = dx.DataFrame({'a': [6.0, 3.0, 3.0, 2.0, 2.0, nan],
                            'b': ['f', 'f', 'd', None, 'er', 'd'],
                            'c': [5.0, 444.0, 1.0, 12.0, 7.0, -5.6]})
        assert_frame_equal(df1, df2)

    def test_sort_values_multiple(self):
        df = dx.DataFrame({'a': [2, 3, nan, 6, 3, 2],
                           'b': [None, 'f', 'd', 'f', 'd', 'er'],
                           'c': [12, 444, -5.6, 5, 1, 7]})

        df1 = df.sort_values(['a', 'b'], ascending=False)
        df2 = dx.DataFrame({'a': [6.0, 3.0, 3.0, 2.0, 2.0, nan],
                            'b': ['f', 'f', 'd', 'er', None, 'd'],
                            'c': [5.0, 444.0, 1.0, 7.0, 12.0, -5.6]})
        assert_frame_equal(df1, df2)

        df1 = df.sort_values(['a', 'b'], ascending=True)
        df2 = dx.DataFrame({'a': [2.0, 2.0, 3.0, 3.0, 6.0, nan],
                            'b': ['er', None, 'd', 'f', 'f', 'd'],
                            'c': [7.0, 12.0, 1.0, 444.0, 5.0, -5.6]})
        assert_frame_equal(df1, df2)

        df1 = df.sort_values(['a', 'b'], ascending=[True, False])
        df2 = dx.DataFrame({'a': [2.0, 2.0, 3.0, 3.0, 6.0, nan],
                            'b': ['er', None, 'f', 'd', 'f', 'd'],
                            'c': [7.0, 12.0, 444.0, 1.0, 5.0, -5.6]})
        assert_frame_equal(df1, df2)

        df1 = df.sort_values(['a', 'b'], ascending=[False, True])
        df2 = dx.DataFrame({'a': [6.0, 3.0, 3.0, 2.0, 2.0, nan],
                            'b': ['f', 'd', 'f', 'er', None, 'd'],
                            'c': [5.0, 1.0, 444.0, 7.0, 12.0, -5.6]})
        assert_frame_equal(df1, df2)

        df1 = df.sort_values(['b', 'a'], ascending=[False, True])
        df2 = dx.DataFrame({'a': [3.0, 6.0, 2.0, 3.0, nan, 2.0],
                            'b': ['f', 'f', 'er', 'd', 'd', None],
                            'c': [444.0, 5.0, 7.0, 1.0, -5.6, 12.0]})
        assert_frame_equal(df1, df2)


class TestRank:

    def test_rank_min(self):
        df = dx.DataFrame({'a': [2, 3, nan, 6, 3, 2],
                           'b': [None, 'f', 'd', 'f', 'd', 'er'],
                           'c': [12, 444, -5.6, 5, 1, 7]})
        df1 = df.rank()
        df2 = dx.DataFrame({'a': [1.0, 3.0, nan, 5.0, 3.0, 1.0],
                            'b': [nan, 4.0, 1.0, 4.0, 1.0, 3.0],
                            'c': [5.0, 6.0, 1.0, 3.0, 2.0, 4.0]})
        assert_frame_equal(df1, df2)

        df1 = df.rank(na_option='top')
        df2 = dx.DataFrame({'a': [2, 4, 1, 6, 4, 2], 'b': [1, 5, 2, 5, 2, 4],
                            'c': [5, 6, 1, 3, 2, 4]})
        assert_frame_equal(df1, df2)

        df1 = df.rank(na_option='bottom')
        df2 = dx.DataFrame({'a': [1, 3, 6, 5, 3, 1], 'b': [6, 4, 1, 4, 1, 3],
                            'c': [5, 6, 1, 3, 2, 4]})
        assert_frame_equal(df1, df2)

        df1 = df.rank(na_option='keep', ascending=False)
        df2 = dx.DataFrame({'a': [4.0, 2.0, nan, 1.0, 2.0, 4.0],
                            'b': [nan, 1.0, 4.0, 1.0, 4.0, 3.0],
                            'c': [2.0, 1.0, 6.0, 4.0, 5.0, 3.0]})
        assert_frame_equal(df1, df2)

        df1 = df.rank(na_option='top', ascending=False)
        df2 = dx.DataFrame({'a': [5, 3, 1, 2, 3, 5], 'b': [1, 2, 5, 2, 5, 4],
                            'c': [2, 1, 6, 4, 5, 3]})
        assert_frame_equal(df1, df2)

        df1 = df.rank(na_option='bottom', ascending=False)
        df2 = dx.DataFrame(
            {'a': [4, 2, 6, 1, 2, 4], 'b': [6, 1, 4, 1, 4, 3], 'c': [2, 1, 6, 4, 5, 3]})
        assert_frame_equal(df1, df2)

    def test_rank_max(self):
        df = dx.DataFrame({'a': [2, 3, nan, 6, 3, 2],
                           'b': [None, 'f', 'd', 'f', 'd', 'er'],
                           'c': [12, 444, -5.6, 5, 1, 7]})

        df1 = df.rank(method='max', na_option='keep', ascending=True)
        df2 = dx.DataFrame({'a': [2.0, 4.0, nan, 5.0, 4.0, 2.0],
                            'b': [nan, 5.0, 2.0, 5.0, 2.0, 3.0],
                            'c': [5.0, 6.0, 1.0, 3.0, 2.0, 4.0]})
        assert_frame_equal(df1, df2)

        df1 = df.rank(method='max', na_option='keep', ascending=False)
        df2 = dx.DataFrame({'a': [5.0, 3.0, nan, 1.0, 3.0, 5.0],
                            'b': [nan, 2.0, 5.0, 2.0, 5.0, 3.0],
                            'c': [2.0, 1.0, 6.0, 4.0, 5.0, 3.0]})
        assert_frame_equal(df1, df2)

        df1 = df.rank(method='max', na_option='top', ascending=True)
        df2 = dx.DataFrame({'a': [3, 5, 1, 6, 5, 3], 'b': [1, 6, 3, 6, 3, 4],
                            'c': [5, 6, 1, 3, 2, 4]})
        assert_frame_equal(df1, df2)

        df1 = df.rank(method='max', na_option='top', ascending=False)
        df2 = dx.DataFrame({'a': [6, 4, 1, 2, 4, 6], 'b': [1, 3, 6, 3, 6, 4],
                            'c': [2, 1, 6, 4, 5, 3]})
        assert_frame_equal(df1, df2)

        df1 = df.rank(method='max', na_option='bottom', ascending=True)
        df2 = dx.DataFrame({'a': [2, 4, 6, 5, 4, 2], 'b': [6, 5, 2, 5, 2, 3],
                            'c': [5, 6, 1, 3, 2, 4]})
        assert_frame_equal(df1, df2)

        df1 = df.rank(method='max', na_option='bottom', ascending=False)
        df2 = dx.DataFrame({'a': [5, 3, 6, 1, 3, 5], 'b': [6, 2, 5, 2, 5, 3],
                            'c': [2, 1, 6, 4, 5, 3]})
        assert_frame_equal(df1, df2)

    def test_rank_dense(self):
        df = dx.DataFrame({'a': [2, 3, nan, 6, 3, 2],
                           'b': [None, 'f', 'd', 'f', 'd', 'er'],
                           'c': [12, 444, -5.6, 5, 1, 7]})

        df1 = df.rank(method='dense', na_option='keep', ascending=True)
        df2 = dx.DataFrame({'a': [1.0, 2.0, nan, 3.0, 2.0, 1.0],
                            'b': [nan, 3.0, 1.0, 3.0, 1.0, 2.0],
                            'c': [5.0, 6.0, 1.0, 3.0, 2.0, 4.0]})
        assert_frame_equal(df1, df2)

        df1 = df.rank(method='dense', na_option='keep', ascending=False)
        df2 = dx.DataFrame({'a': [3.0, 2.0, nan, 1.0, 2.0, 3.0],
                            'b': [nan, 1.0, 3.0, 1.0, 3.0, 2.0],
                            'c': [2.0, 1.0, 6.0, 4.0, 5.0, 3.0]})
        assert_frame_equal(df1, df2)

        df1 = df.rank(method='dense', na_option='top', ascending=True)
        df2 = dx.DataFrame({'a': [2, 3, 1, 4, 3, 2], 'b': [1, 4, 2, 4, 2, 3],
                            'c': [5, 6, 1, 3, 2, 4]})
        assert_frame_equal(df1, df2)

        df1 = df.rank(method='dense', na_option='top', ascending=False)
        df2 = dx.DataFrame(
            {'a': [4, 3, 1, 2, 3, 4], 'b': [1, 2, 4, 2, 4, 3], 'c': [2, 1, 6, 4, 5, 3]})
        assert_frame_equal(df1, df2)

        df1 = df.rank(method='dense', na_option='bottom', ascending=True)
        df2 = dx.DataFrame(
            {'a': [1, 2, 4, 3, 2, 1], 'b': [4, 3, 1, 3, 1, 2], 'c': [5, 6, 1, 3, 2, 4]})
        assert_frame_equal(df1, df2)

        df1 = df.rank(method='dense', na_option='bottom', ascending=False)
        df2 = dx.DataFrame(
            {'a': [3, 2, 4, 1, 2, 3], 'b': [4, 1, 3, 1, 3, 2], 'c': [2, 1, 6, 4, 5, 3]})
        assert_frame_equal(df1, df2)

    def test_rank_first(self):
        df = dx.DataFrame({'a': [2, 3, nan, 6, 3, 2],
                           'b': [None, 'f', 'd', 'f', 'd', 'er'],
                           'c': [12, 444, -5.6, 5, 1, 7]})

        df1 = df.rank(method='first', na_option='keep', ascending=True)
        df2 = dx.DataFrame({'a': [1.0, 3.0, nan, 5.0, 4.0, 2.0],
                            'b': [nan, 4.0, 1.0, 5.0, 2.0, 3.0],
                            'c': [5.0, 6.0, 1.0, 3.0, 2.0, 4.0]})
        assert_frame_equal(df1, df2)

        df1 = df.rank(method='first', na_option='keep', ascending=False)
        df2 = dx.DataFrame({'a': [4.0, 2.0, nan, 1.0, 3.0, 5.0],
                            'b': [nan, 1.0, 4.0, 2.0, 5.0, 3.0],
                            'c': [2.0, 1.0, 6.0, 4.0, 5.0, 3.0]})
        assert_frame_equal(df1, df2)

        df1 = df.rank(method='first', na_option='top', ascending=True)
        df2 = dx.DataFrame(
            {'a': [2, 4, 1, 6, 5, 3], 'b': [1, 5, 2, 6, 3, 4], 'c': [5, 6, 1, 3, 2, 4]})
        assert_frame_equal(df1, df2)

        df1 = df.rank(method='first', na_option='top', ascending=False)
        df2 = dx.DataFrame(
            {'a': [5, 3, 1, 2, 4, 6], 'b': [1, 2, 5, 3, 6, 4], 'c': [2, 1, 6, 4, 5, 3]})
        assert_frame_equal(df1, df2)

        df1 = df.rank(method='first', na_option='bottom', ascending=True)
        df2 = dx.DataFrame(
            {'a': [1, 3, 6, 5, 4, 2], 'b': [6, 4, 1, 5, 2, 3], 'c': [5, 6, 1, 3, 2, 4]})
        assert_frame_equal(df1, df2)

        df1 = df.rank(method='first', na_option='bottom', ascending=False)
        df2 = dx.DataFrame(
            {'a': [4, 2, 6, 1, 3, 5], 'b': [6, 1, 4, 2, 5, 3], 'c': [2, 1, 6, 4, 5, 3]})
        assert_frame_equal(df1, df2)

    def test_rank_average(self):
        df = dx.DataFrame({'a': [2, 3, nan, 6, 3, 2],
                           'b': [None, 'f', 'd', 'f', 'd', 'er'],
                           'c': [12, 444, -5.6, 5, 1, 7]})

        df1 = df.rank(method='average', na_option='keep', ascending=True)
        df2 = dx.DataFrame({'a': [1.5, 3.5, nan, 5.0, 3.5, 1.5],
                            'b': [nan, 4.5, 1.5, 4.5, 1.5, 3.0],
                            'c': [5.0, 6.0, 1.0, 3.0, 2.0, 4.0]})
        assert_frame_equal(df1, df2)

        df1 = df.rank(method='average', na_option='keep', ascending=False)
        df2 = dx.DataFrame({'a': [4.5, 2.5, nan, 1.0, 2.5, 4.5],
                            'b': [nan, 1.5, 4.5, 1.5, 4.5, 3.0],
                            'c': [2.0, 1.0, 6.0, 4.0, 5.0, 3.0]})
        assert_frame_equal(df1, df2)

        df1 = df.rank(method='average', na_option='top', ascending=True)
        df2 = dx.DataFrame({'a': [2.5, 4.5, 1.0, 6.0, 4.5, 2.5],
                            'b': [1.0, 5.5, 2.5, 5.5, 2.5, 4.0],
                            'c': [5.0, 6.0, 1.0, 3.0, 2.0, 4.0]})
        assert_frame_equal(df1, df2)

        df1 = df.rank(method='average', na_option='top', ascending=False)
        df2 = dx.DataFrame({'a': [5.5, 3.5, 1.0, 2.0, 3.5, 5.5],
                            'b': [1.0, 2.5, 5.5, 2.5, 5.5, 4.0],
                            'c': [2.0, 1.0, 6.0, 4.0, 5.0, 3.0]})
        assert_frame_equal(df1, df2)

        df1 = df.rank(method='average', na_option='bottom', ascending=True)
        df2 = dx.DataFrame({'a': [1.5, 3.5, 6.0, 5.0, 3.5, 1.5],
                            'b': [6.0, 4.5, 1.5, 4.5, 1.5, 3.0],
                            'c': [5.0, 6.0, 1.0, 3.0, 2.0, 4.0]})
        assert_frame_equal(df1, df2)

        df1 = df.rank(method='average', na_option='bottom', ascending=False)
        df2 = dx.DataFrame({'a': [4.5, 2.5, 6.0, 1.0, 2.5, 4.5],
                            'b': [6.0, 1.5, 4.5, 1.5, 4.5, 3.0],
                            'c': [2.0, 1.0, 6.0, 4.0, 5.0, 3.0]})
        assert_frame_equal(df1, df2)


class TestStreak:

    def test_streak(self):
        df = dx.DataFrame(
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
        df = dx.DataFrame(
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
        df = dx.DataFrame(
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
        df = dx.DataFrame(data)

        df1 = df.drop(columns='b')
        df2 = dx.DataFrame({'a': [0, 0, 5, 9],
                            'c': ['', 'e', 'f', 's'],
                            'd': [False, False, True, False],
                            'e': [0, 20, 30, 4],
                            'f': ['a', None, 'ad', 'effd'],
                            'g': [nan, nan, nan, nan]})
        assert_frame_equal(df1, df2)

        df1 = df.drop(columns='g')
        df2 = dx.DataFrame({'a': [0, 0, 5, 9],
                            'b': [0.0, 1.5, 8.0, 9.0],
                            'c': ['', 'e', 'f', 's'],
                            'd': [False, False, True, False],
                            'e': [0, 20, 30, 4],
                            'f': ['a', None, 'ad', 'effd']})
        assert_frame_equal(df1, df2)

        df1 = df.drop(columns='d')
        df2 = dx.DataFrame({'a': [0, 0, 5, 9],
                            'b': [0.0, 1.5, 8.0, 9.0],
                            'c': ['', 'e', 'f', 's'],
                            'e': [0, 20, 30, 4],
                            'f': ['a', None, 'ad', 'effd'],
                            'g': [nan, nan, nan, nan]})
        assert_frame_equal(df1, df2)

        df1 = df.drop(columns=list('abcdefg'))
        data = {'i': np.empty((4, 0), dtype=np.int64),
                'f': array((4, 0), dtype=np.float64),
                'O': array((4, 0), dtype=np.object),
                'b': array((4, 0), dtype=np.bool)}
        column_info = {}
        columns = []
        df2 = df1._construct_from_new(data, column_info, columns)
        assert_frame_equal(df1, df2)

        df1 = df.drop(columns=list('ade'))
        df2 = dx.DataFrame({'b': [0.0, 1.5, 8.0, 9.0],
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
        df = dx.DataFrame(data)

        df1 = df.drop(rows=3)
        df2 = dx.DataFrame({'a': [0, 0, 5],
                            'b': [0.0, 1.5, 8.0],
                            'c': ['', 'e', 'f'],
                            'd': [False, False, True],
                            'e': [0, 20, 30],
                            'f': ['a', None, 'ad'],
                            'g': [nan, nan, nan]})
        assert_frame_equal(df1, df2)

        with pytest.raises(IndexError):
            df.drop(rows=5)

        with pytest.raises(IndexError):
            df.drop(rows=[-1, 0, -3])

    def test_drop_rows_and_cols(self):
        data = {'a': [0, 0, 5, 9],
                'b': [0, 1.5, 8, 9],
                'c': [''] + list('efs'),
                'd': [False, False, True, False],
                'e': [0, 20, 30, 4],
                'f': ['a', nan, 'ad', 'effd'],
                'g': [np.nan] * 4}
        df = dx.DataFrame(data)

        df1 = df.drop(1, 1)
        df2 = dx.DataFrame({'a': [0, 5, 9],
                            'c': ['', 'f', 's'],
                            'd': [False, True, False],
                            'e': [0, 30, 4],
                            'f': ['a', 'ad', 'effd'],
                            'g': [nan, nan, nan]})
        assert_frame_equal(df1, df2)

        df1 = df.drop(-2, list('abc'))
        df2 = dx.DataFrame({'d': [False, False, False],
                            'e': [0, 20, 4],
                            'f': ['a', None, 'effd'],
                            'g': [nan, nan, nan]})
        assert_frame_equal(df1, df2)

        df1 = df.drop([0, 3], [3, 'a', -2])
        df2 = dx.DataFrame({'b': [1.5, 8.0], 'c': ['e', 'f'], 'e': [20, 30], 'g': [nan, nan]})
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
        df = dx.DataFrame(data)

        with pytest.raises(ValueError):
            df.rename({'a': 'b'})

        df1 = df.rename({'a': 'alpha'})
        df2 = dx.DataFrame({'alpha': [0, 0, 5, 9],
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
        df2 = dx.DataFrame({'b': [0, 0, 5, 9],
                            't': [0.0, 1.5, 8.0, 9.0],
                            'c': ['', 'e', 'f', 's'],
                            'd': [False, False, True, False],
                            'e': [0, 20, 30, 4],
                            'f': ['a', None, 'ad', 'effd'],
                            'g': [nan, nan, nan, nan]})
        assert_frame_equal(df1, df2)

        df1 = df.rename(list('poiuytr'))
        df2 = dx.DataFrame({'p': [0, 0, 5, 9], 'o': [0.0, 1.5, 8.0, 9.0], 'i': ['', 'e', 'f', 's'],
                            'u': [False, False, True, False], 'y': [0, 20, 30, 4],
                            't': ['a', None, 'ad', 'effd'], 'r': [nan, nan, nan, nan]})
        assert_frame_equal(df1, df2)


class TestNLargest:

    def test_nlargest_int(self):
        data = {'a': [9, 10, 9, 9, 10],
                'b': [0, nan, nan, 0, 1],
                'c': [''] + list('eeaz'),
                'd': [False, False, True, False, True],
                'e': [0, 20, 30, 4, 4],
                'f': ['a', nan, 'ad', None, 'ad'],
                'g': [np.nan] * 5}
        df = dx.DataFrame(data)

        df1 = df.nlargest(3, 'a', keep='all')
        df2 = dx.DataFrame({'a': [10, 10, 9, 9, 9],
                            'b': [nan, 1.0, 0.0, nan, 0.0],
                            'c': ['e', 'z', '', 'e', 'a'],
                            'd': [False, True, False, True, False],
                            'e': [20, 4, 0, 30, 4],
                            'f': [None, 'ad', 'a', 'ad', None],
                            'g': [nan, nan, nan, nan, nan]})
        assert_frame_equal(df1, df2)

        df1 = df.nlargest(3, 'a', keep='first')
        df2 = dx.DataFrame({'a': [10, 10, 9],
                            'b': [nan, 1.0, 0.0],
                            'c': ['e', 'z', ''],
                            'd': [False, True, False],
                            'e': [20, 4, 0],
                            'f': [None, 'ad', 'a'],
                            'g': [nan, nan, nan]})
        assert_frame_equal(df1, df2)

        df1 = df.nlargest(3, 'a', keep='last')
        df2 = dx.DataFrame({'a': [10, 10, 9],
                            'b': [nan, 1.0, 0.0],
                            'c': ['e', 'z', 'a'],
                            'd': [False, True, False],
                            'e': [20, 4, 4],
                            'f': [None, 'ad', None],
                            'g': [nan, nan, nan]})
        assert_frame_equal(df1, df2)

        df1 = df.nlargest(10, 'a', keep='last')
        df2 = dx.DataFrame({'a': [10, 10, 9, 9, 9],
                            'b': [nan, 1.0, 0.0, nan, 0.0],
                            'c': ['e', 'z', '', 'e', 'a'],
                            'd': [False, True, False, True, False],
                            'e': [20, 4, 0, 30, 4],
                            'f': [None, 'ad', 'a', 'ad', None],
                            'g': [nan, nan, nan, nan, nan]})
        assert_frame_equal(df1, df2)

    def test_nlargest_float(self):
        data = {'a': [9, 10, 9, 9, 10],
                'b': [0, nan, nan, 0, 1],
                'c': [''] + list('eeaz'),
                'd': [False, False, True, False, True],
                'e': [0, 20, 30, 4, 4],
                'f': ['a', nan, 'ad', None, 'ad'],
                'g': [np.nan] * 5}
        df = dx.DataFrame(data)

        df1 = df.nlargest(2, 'b')
        df2 = dx.DataFrame({'a': [10, 9, 9],
                            'b': [1.0, 0.0, 0.0],
                            'c': ['z', '', 'a'],
                            'd': [True, False, False],
                            'e': [4, 0, 4],
                            'f': ['ad', 'a', None],
                            'g': [nan, nan, nan]})
        assert_frame_equal(df1, df2)

        df1 = df.nlargest(2, 'b', keep='first')
        df2 = dx.DataFrame({'a': [10, 9],
                            'b': [1.0, 0.0],
                            'c': ['z', ''],
                            'd': [True, False],
                            'e': [4, 0],
                            'f': ['ad', 'a'],
                            'g': [nan, nan]})
        assert_frame_equal(df1, df2)

        df1 = df.nlargest(2, 'b', keep='last')
        df2 = dx.DataFrame({'a': [10, 9],
                            'b': [1.0, 0.0],
                            'c': ['z', 'a'],
                            'd': [True, False],
                            'e': [4, 4],
                            'f': ['ad', None],
                            'g': [nan, nan]})
        assert_frame_equal(df1, df2)

        df1 = df.nlargest(1, 'g', keep='all')
        df2 = dx.DataFrame({'a': [9],
                            'b': [0.0],
                            'c': [''],
                            'd': [False],
                            'e': [0],
                            'f': ['a'],
                            'g': [nan]})
        assert_frame_equal(df1, df2)

    def test_nlargest_str(self):
        data = {'a': [9, 10, 9, 9, 10],
                'b': [0, nan, nan, 0, 1],
                'c': [''] + list('eeaz'),
                'd': [False, False, True, False, True],
                'e': [0, 20, 30, 4, 4],
                'f': ['a', nan, 'ad', None, 'ad'],
                'g': [np.nan] * 5}
        df = dx.DataFrame(data)

        df1 = df.nlargest(2, 'c', keep='all')
        df2 = dx.DataFrame({'a': [10, 10, 9],
                            'b': [1.0, nan, nan],
                            'c': ['z', 'e', 'e'],
                            'd': [True, False, True],
                            'e': [4, 20, 30],
                            'f': ['ad', None, 'ad'],
                            'g': [nan, nan, nan]})
        assert_frame_equal(df1, df2)

        df1 = df.nlargest(2, 'c', keep='first')
        df2 = dx.DataFrame({'a': [10, 10],
                            'b': [1.0, nan],
                            'c': ['z', 'e'],
                            'd': [True, False],
                            'e': [4, 20],
                            'f': ['ad', None],
                            'g': [nan, nan]})
        assert_frame_equal(df1, df2)

        df1 = df.nlargest(2, 'c', keep='last')
        df2 = dx.DataFrame({'a': [10, 9],
                            'b': [1.0, nan],
                            'c': ['z', 'e'],
                            'd': [True, True],
                            'e': [4, 30],
                            'f': ['ad', 'ad'],
                            'g': [nan, nan]})
        assert_frame_equal(df1, df2)

    def test_nlargest_bool(self):
        data = {'a': [9, 10, 9, 9, 10],
                'b': [0, nan, nan, 0, 1],
                'c': [''] + list('eeaz'),
                'd': [False, False, True, False, True],
                'e': [0, 20, 30, 4, 4],
                'f': ['a', nan, 'ad', None, 'ad'],
                'g': [np.nan] * 5}
        df = dx.DataFrame(data)

        df1 = df.nlargest(3, 'd', keep='all')
        df2 = dx.DataFrame({'a': [9, 10, 9, 10, 9],
                            'b': [nan, 1.0, 0.0, nan, 0.0],
                            'c': ['e', 'z', '', 'e', 'a'],
                            'd': [True, True, False, False, False],
                            'e': [30, 4, 0, 20, 4],
                            'f': ['ad', 'ad', 'a', None, None],
                            'g': [nan, nan, nan, nan, nan]})
        assert_frame_equal(df1, df2)

        df1 = df.nlargest(3, 'd', keep='first')
        df2 = dx.DataFrame({'a': [9, 10, 9],
                            'b': [nan, 1.0, 0.0],
                            'c': ['e', 'z', ''],
                            'd': [True, True, False],
                            'e': [30, 4, 0],
                            'f': ['ad', 'ad', 'a'],
                            'g': [nan, nan, nan]})
        assert_frame_equal(df1, df2)

        df1 = df.nlargest(3, 'd', keep='last')
        df2 = dx.DataFrame({'a': [9, 10, 9],
                            'b': [nan, 1.0, 0.0],
                            'c': ['e', 'z', 'a'],
                            'd': [True, True, False],
                            'e': [30, 4, 4],
                            'f': ['ad', 'ad', None],
                            'g': [nan, nan, nan]})
        assert_frame_equal(df1, df2)

    def test_nsmallest_int(self):
        data = {'a': [9, 10, 9, 9, 10],
                'b': [0, nan, nan, 0, 1],
                'c': [''] + list('eeaz'),
                'd': [False, False, True, False, True],
                'e': [0, 20, 30, 4, 4],
                'f': ['a', nan, 'ad', None, 'ad'],
                'g': [np.nan] * 5}
        df = dx.DataFrame(data)

        df1 = df.nsmallest(2, 'a', keep='all')
        df2 = dx.DataFrame({'a': [9, 9, 9],
                            'b': [0.0, nan, 0.0],
                            'c': ['', 'e', 'a'],
                            'd': [False, True, False],
                            'e': [0, 30, 4],
                            'f': ['a', 'ad', None],
                            'g': [nan, nan, nan]})
        assert_frame_equal(df1, df2)

        df1 = df.nsmallest(2, 'a', keep='first')
        df2 = dx.DataFrame({'a': [9, 9],
                            'b': [0.0, nan],
                            'c': ['', 'e'],
                            'd': [False, True],
                            'e': [0, 30],
                            'f': ['a', 'ad'],
                            'g': [nan, nan]})
        assert_frame_equal(df1, df2)

        df1 = df.nsmallest(2, 'a', keep='last')
        df2 = dx.DataFrame({'a': [9, 9],
                            'b': [nan, 0.0],
                            'c': ['e', 'a'],
                            'd': [True, False],
                            'e': [30, 4],
                            'f': ['ad', None],
                            'g': [nan, nan]})
        assert_frame_equal(df1, df2)

    def test_nsmallest_float(self):
        data = {'a': [9, 10, 9, 9, 10],
                'b': [0, nan, nan, 0, 1],
                'c': [''] + list('eeaz'),
                'd': [False, False, True, False, True],
                'e': [0, 20, 30, 4, 4],
                'f': ['a', nan, 'ad', None, 'ad'],
                'g': [np.nan] * 5}
        df = dx.DataFrame(data)

        df1 = df.nsmallest(1, 'b')
        df2 = dx.DataFrame({'a': [9, 9],
                            'b': [0.0, 0.0],
                            'c': ['', 'a'],
                            'd': [False, False],
                            'e': [0, 4],
                            'f': ['a', None],
                            'g': [nan, nan]})
        assert_frame_equal(df1, df2)

        df1 = df.nsmallest(1, 'b', keep='first')
        df2 = dx.DataFrame({'a': [9],
                            'b': [0.0],
                            'c': [''],
                            'd': [False],
                            'e': [0],
                            'f': ['a'],
                            'g': [nan]})
        assert_frame_equal(df1, df2)

        df1 = df.nsmallest(1, 'b', keep='last')
        df2 = dx.DataFrame({'a': [9],
                            'b': [0.0],
                            'c': ['a'],
                            'd': [False],
                            'e': [4],
                            'f': [None],
                            'g': [nan]})
        assert_frame_equal(df1, df2)

    def test_nsmallest_str(self):
        data = {'a': [9, 10, 9, 9, 10],
                'b': [0, nan, nan, 0, 1],
                'c': [''] + list('eeaz'),
                'd': [False, False, True, False, True],
                'e': [0, 20, 30, 4, 4],
                'f': ['a', nan, 'ad', None, 'ad'],
                'g': [np.nan] * 5}
        df = dx.DataFrame(data)

        df1 = df.nsmallest(3, 'c', keep='all')
        df2 = dx.DataFrame({'a': [9, 9, 10, 9],
                            'b': [0.0, 0.0, nan, nan],
                            'c': ['', 'a', 'e', 'e'],
                            'd': [False, False, False, True],
                            'e': [0, 4, 20, 30],
                            'f': ['a', None, None, 'ad'],
                            'g': [nan, nan, nan, nan]})
        assert_frame_equal(df1, df2)

        df1 = df.nsmallest(3, 'c', keep='first')
        df2 = dx.DataFrame({'a': [9, 9, 10],
                            'b': [0.0, 0.0, nan],
                            'c': ['', 'a', 'e'],
                            'd': [False, False, False],
                            'e': [0, 4, 20],
                            'f': ['a', None, None],
                            'g': [nan, nan, nan]})
        assert_frame_equal(df1, df2)

        df1 = df.nsmallest(3, 'c', keep='last')
        df2 = dx.DataFrame({'a': [9, 9, 9],
                            'b': [0.0, 0.0, nan],
                            'c': ['', 'a', 'e'],
                            'd': [False, False, True],
                            'e': [0, 4, 30],
                            'f': ['a', None, 'ad'],
                            'g': [nan, nan, nan]})
        assert_frame_equal(df1, df2)

    def test_nsmallest_bool(self):
        data = {'a': [9, 10, 9, 9, 10],
                'b': [0, nan, nan, 0, 1],
                'c': [''] + list('eeaz'),
                'd': [False, False, True, False, True],
                'e': [0, 20, 30, 4, 4],
                'f': ['a', nan, 'ad', None, 'ad'],
                'g': [np.nan] * 5}
        df = dx.DataFrame(data)

        df1 = df.nsmallest(2, 'd', keep='all')
        df2 = dx.DataFrame({'a': [9, 10, 9],
                            'b': [0.0, nan, 0.0],
                            'c': ['', 'e', 'a'],
                            'd': [False, False, False],
                            'e': [0, 20, 4],
                            'f': ['a', None, None],
                            'g': [nan, nan, nan]})
        assert_frame_equal(df1, df2)

        df1 = df.nsmallest(2, 'd', keep='first')
        df2 = dx.DataFrame({'a': [9, 10],
                            'b': [0.0, nan],
                            'c': ['', 'e'],
                            'd': [False, False],
                            'e': [0, 20],
                            'f': ['a', None],
                            'g': [nan, nan]})
        assert_frame_equal(df1, df2)

        df1 = df.nsmallest(2, 'd', keep='last')
        df2 = dx.DataFrame({'a': [10, 9],
                            'b': [nan, 0.0],
                            'c': ['e', 'a'],
                            'd': [False, False],
                            'e': [20, 4],
                            'f': [None, None],
                            'g': [nan, nan]})
        assert_frame_equal(df1, df2)


class TestFactorize:

    def test_factorize(self):
        data = {'a': [9, 10, 9, 9, 10],
                'b': [0, nan, nan, 0, 1],
                'c': [''] + list('eeaz'),
                'd': [False, False, True, False, True],
                'e': [0, 20, 30, 4, 4],
                'f': ['a', nan, 'ad', None, 'ad'],
                'g': [np.nan] * 5}
        df = dx.DataFrame(data)

        arr11, arr12 = df.factorize('a')
        arr21, arr22 = (array([0, 1, 0, 0, 1]), array([9, 10]))

        assert_array_equal(arr11, arr21)
        assert_array_equal(arr12, arr22)

        arr11, arr12 = df.factorize('b')
        arr21, arr22 = (array([0, 1, 1, 0, 2]), array([0., nan, 1.]))

        assert_array_equal(arr11, arr21)
        assert_array_equal(arr12, arr22)

        arr11, arr12 = df.factorize('c')
        arr21, arr22 = (array([0, 1, 1, 2, 3]), array(['', 'e', 'a', 'z'], dtype=object))

        assert_array_equal(arr11, arr21)
        assert_array_equal(arr12, arr22)

        arr11, arr12 = df.factorize('d')
        arr21, arr22 = (array([0, 0, 1, 0, 1]), array([False, True]))

        assert_array_equal(arr11, arr21)
        assert_array_equal(arr12, arr22)

    def test_sample(self):
        data = {'a': [9, 10, 9, 9, 10],
                'b': [0, nan, nan, 0, 1],
                'c': [''] + list('eeaz'),
                'd': [False, False, True, False, True],
                'e': [0, 20, 30, 4, 4],
                'f': ['a', nan, 'ad', None, 'ad'],
                'g': [np.nan] * 5}
        df = dx.DataFrame(data)

        df1 = df.sample(2, random_state=1)
        df2 = dx.DataFrame({'a': [9, 10],
                            'b': [nan, nan],
                            'c': ['e', 'e'],
                            'd': [True, False],
                            'e': [30, 20],
                            'f': ['ad', None],
                            'g': [nan, nan]})
        assert_frame_equal(df1, df2)

        with pytest.raises(ValueError):
            df.sample(10, random_state=1)

        with pytest.raises(ValueError):
            df.sample(3, frac=.3, random_state=1)

        with pytest.raises(ValueError):
            df.sample(frac=-1, random_state=1)

        df1 = df.sample(frac=.5, random_state=1)
        df2 = dx.DataFrame({'a': [9, 10, 10],
                            'b': [nan, nan, 1.0],
                            'c': ['e', 'e', 'z'],
                            'd': [True, False, True],
                            'e': [30, 20, 4],
                            'f': ['ad', None, 'ad'],
                            'g': [nan, nan, nan]})
        assert_frame_equal(df1, df2)

        df1 = df.sample(12, random_state=1, replace=True)
        df2 = dx.DataFrame({'a': [9, 10, 9, 10, 9, 9, 9, 10, 10, 10, 10, 9],
                            'b': [0.0, 1.0, 0.0, nan, 0.0, 0.0, 0.0, nan, 1.0, 1.0, nan, nan],
                            'c': ['a', 'z', '', 'e', 'a', '', '', 'e', 'z', 'z', 'e', 'e'],
                            'd': [False, True, False, False, False, False, False, False, True, True,
                                  False, True],
                            'e': [4, 4, 0, 20, 4, 0, 0, 20, 4, 4, 20, 30],
                            'f': [None, 'ad', 'a', None, None, 'a', 'a', None, 'ad', 'ad', None,
                                  'ad'],
                            'g': [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]})
        assert_frame_equal(df1, df2)

        df1 = df.sample(12, random_state=1, replace=True, weights=[1, 5, 10, 3, 4])
        df2 = dx.DataFrame({'a': [9, 9, 9, 9, 10, 10, 10, 9, 9, 9, 9, 9],
                            'b': [nan, 0.0, 0.0, nan, nan, nan, nan, nan, nan, nan, nan, nan],
                            'c': ['e', 'a', '', 'e', 'e', 'e', 'e', 'e', 'e', 'e', 'e', 'e'],
                            'd': [True, False, False, True, False, False, False, True, True, True,
                                  True, True], 'e': [30, 4, 0, 30, 20, 20, 20, 30, 30, 30, 30, 30],
                            'f': ['ad', None, 'a', 'ad', None, None, None, 'ad', 'ad', 'ad', 'ad',
                                  'ad'],
                            'g': [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]})
        assert_frame_equal(df1, df2)


class TestIsIn:

    def test_is_in_scalar_list(self):
        data = {'a': [9, 10, 9, 9, 10],
                'b': [0, nan, nan, 0, 1],
                'c': [''] + list('eeaz'),
                'd': [False, False, True, False, True],
                'e': [0, 20, 30, 4, 4],
                'f': ['a', nan, 'ad', None, 'ad'],
                'g': [np.nan] * 5}
        df = dx.DataFrame(data)

        df1 = df.isin(0)
        df2 = dx.DataFrame({'a': [False, False, False, False, False],
                            'b': [True, False, False, True, False],
                            'c': [False, False, False, False, False],
                            'd': [True, True, False, True, False],
                            'e': [True, False, False, False, False],
                            'f': [False, False, False, False, False],
                            'g': [False, False, False, False, False]})
        assert_frame_equal(df1, df2)

        df1 = df.isin([9, 20])
        df2 = dx.DataFrame({'a': [True, False, True, True, False],
                            'b': [False, False, False, False, False],
                            'c': [False, False, False, False, False],
                            'd': [False, False, False, False, False],
                            'e': [False, True, False, False, False],
                            'f': [False, False, False, False, False],
                            'g': [False, False, False, False, False]})
        assert_frame_equal(df1, df2)

        df1 = df.isin([10, 55, 'e'])
        df2 = dx.DataFrame({'a': [False, True, False, False, True],
                            'b': [False, False, False, False, False],
                            'c': [False, True, True, False, False],
                            'd': [False, False, False, False, False],
                            'e': [False, False, False, False, False],
                            'f': [False, False, False, False, False],
                            'g': [False, False, False, False, False]})
        assert_frame_equal(df1, df2)

    def test_isin_dict(self):
        data = {'a': [9, 10, 9, 9, 10],
                'b': [0, nan, nan, 0, 1],
                'c': [''] + list('eeaz'),
                'd': [False, False, True, False, True],
                'e': [0, 20, 30, 4, 4],
                'f': ['a', nan, 'ad', None, 'ad'],
                'g': [np.nan] * 5}
        df = dx.DataFrame(data)

        df1 = df.isin({'b': 0})
        df2 = dx.DataFrame({'a': [False, False, False, False, False],
                            'b': [True, False, False, True, False],
                            'c': [False, False, False, False, False],
                            'd': [False, False, False, False, False],
                            'e': [False, False, False, False, False],
                            'f': [False, False, False, False, False],
                            'g': [False, False, False, False, False]})
        assert_frame_equal(df1, df2)

        df1 = df.isin({'b': [0, 9, 'e', True]})
        df2 = dx.DataFrame({'a': [False, False, False, False, False],
                            'b': [True, False, False, True, True],
                            'c': [False, False, False, False, False],
                            'd': [False, False, False, False, False],
                            'e': [False, False, False, False, False],
                            'f': [False, False, False, False, False],
                            'g': [False, False, False, False, False]})
        assert_frame_equal(df1, df2)

        df1 = df.isin({'b': [0, 9, 'e', True], 'c': ['e', 5, 'z']})
        df2 = dx.DataFrame({'a': [False, False, False, False, False],
                            'b': [True, False, False, True, True],
                            'c': [False, True, True, False, True],
                            'd': [False, False, False, False, False],
                            'e': [False, False, False, False, False],
                            'f': [False, False, False, False, False],
                            'g': [False, False, False, False, False]})
        assert_frame_equal(df1, df2)


class TestWhere:

    def test_where_numeric_cols(self):
        data = {'a': [9, 10, 9, 9, 10],
                'b': [0, nan, nan, 0, 1],
                'c': [''] + list('eeaz'),
                'd': [False, False, True, False, True],
                'e': [0, 20, 30, 4, 4],
                'f': ['a', nan, 'ad', None, 'ad'],
                'g': [np.nan] * 5}
        df = dx.DataFrame(data)

        cond = df[:, 'e'] > 9
        df1 = df[:, ['a', 'b', 'd', 'e']].where(cond)
        df2 = dx.DataFrame({'a': [nan, 10.0, 9.0, nan, nan],
                            'b': [nan, nan, nan, nan, nan],
                            'd': [nan, 0.0, 1.0, nan, nan],
                            'e': [nan, 20.0, 30.0, nan, nan]})
        assert_frame_equal(df1, df2)

        df1 = df[:, ['a', 'b', 'e']].where(cond, 22)
        df2 = dx.DataFrame({'a': [nan, 22.0, 22.0, nan, nan],
                            'b': [nan, 22.0, 22.0, nan, nan],
                            'e': [nan, 22.0, 22.0, nan, nan]})
        assert_frame_equal(df1, df2)

        df1 = df[:, ['a', 'b', 'e']].where(cond, 22, 99)
        df2 = dx.DataFrame({'a': [99, 22, 22, 99, 99],
                            'b': [99, 22, 22, 99, 99],
                            'e': [99, 22, 22, 99, 99]})
        assert_frame_equal(df1, df2)

        df1 = df.where(cond)
        df2 = dx.DataFrame({'a': [nan, 10.0, 9.0, nan, nan],
                            'b': [nan, nan, nan, nan, nan],
                            'c': [None, 'e', 'e', None, None],
                            'd': [nan, 0.0, 1.0, nan, nan],
                            'e': [nan, 20.0, 30.0, nan, nan],
                            'f': [None, None, 'ad', None, None],
                            'g': [nan, nan, nan, nan, nan]})
        assert_frame_equal(df1, df2)

    def test_where_string_cols(self):
        data = {'a': [9, 10, 9, 9, 10],
                'b': [0, nan, nan, 0, 1],
                'c': [''] + list('eeaz'),
                'd': [False, False, True, False, True],
                'e': [0, 20, 30, 4, 4],
                'f': ['a', nan, 'ad', None, 'ad'],
                'g': [np.nan] * 5}
        df = dx.DataFrame(data)
        cond = df[:, 'e'] > 9

        df1 = df[:, ['c', 'f']].where(cond)
        df2 = dx.DataFrame({'c': [None, 'e', 'e', None, None],
                            'f': [None, None, 'ad', None, None]})
        assert_frame_equal(df1, df2)

        df1 = df[:, ['c', 'f']].where(cond, 22, 99)
        df2 = dx.DataFrame({'c': [99, 22, 22, 99, 99], 'f': [99, 22, 22, 99, 99]})
        assert_frame_equal(df1, df2)

        df1 = df[:, ['c', 'f']].where(cond, 't')
        df2 = dx.DataFrame({'c': [None, 't', 't', None, None], 'f': [None, 't', 't', None, None]})
        assert_frame_equal(df1, df2)

        df1 = df[:, ['c', 'f']].where(cond, 't', 'y')
        df2 = dx.DataFrame({'c': ['y', 't', 't', 'y', 'y'], 'f': ['y', 't', 't', 'y', 'y']})
        assert_frame_equal(df1, df2)

    def test_where_array_xy(self):
        data = {'a': [9, 10, 9, 9, 10],
                'b': [0, nan, nan, 0, 1],
                'c': [''] + list('eeaz'),
                'd': [False, False, True, False, True],
                'e': [0, 20, 30, 4, 4],
                'f': ['a', nan, 'ad', None, 'ad'],
                'g': [np.nan] * 5}
        df = dx.DataFrame(data)
        cond = df[:, 'e'] > 9

        df1 = df[:, ['c', 'f']].where(cond, np.arange(5), np.arange(10, 15))
        df2 = dx.DataFrame({'c': [10, 1, 2, 13, 14], 'f': [10, 1, 2, 13, 14]})
        assert_frame_equal(df1, df2)

        df1 = df[:, ['c', 'f']].where(cond, np.arange(5), 99)
        df2 = dx.DataFrame({'c': [99, 1, 2, 99, 99], 'f': [99, 1, 2, 99, 99]})
        assert_frame_equal(df1, df2)

        with pytest.raises(TypeError):
            df[:, ['c', 'f']].where(cond, np.arange(5), 'er')

        df1 = df[:, ['c', 'f']].where(cond, y='er')
        df2 = dx.DataFrame({'c': ['er', 'e', 'e', 'er', 'er'], 'f': ['er', None, 'ad', 'er', 'er']})
        assert_frame_equal(df1, df2)


class TestAppend:

    data = {'a': [9, 10, 9, 9, 10],
            'b': [0, nan, nan, 0, 1],
            'c': [None] + list('eeaz'),
            'd': [False, False, True, False, True],
            'e': [0, 20, 30, 4, 4],
            'f': ['a', nan, 'ad', None, 'ad'],
            'g': [np.nan] * 5}
    df = dx.DataFrame(data)

    def test_append_scalar_column(self):
        df1 = self.df.append({'h': 10}, axis='columns')
        data2 = {'a': array([ 9, 10,  9,  9, 10]),
                 'b': array([ 0., nan, nan,  0.,  1.]),
                 'c': array([None, 'e', 'e', 'a', 'z'], dtype=object),
                 'd': array([False, False,  True, False,  True]),
                 'e': array([ 0, 20, 30,  4,  4]),
                 'f': array(['a', None, 'ad', None, 'ad'], dtype=object),
                 'g': array([nan, nan, nan, nan, nan]),
                 'h': array([10, 10, 10, 10, 10])}
        df2 = dx.DataFrame(data2)
        assert_frame_equal(df1, df2)

    def test_append_multiple_scalar_columns(self):
        df1 = self.df.append({'h': 10, 'i': 4.2, 'j': 'asdf'}, axis='columns')
        data2 = {'a': array([ 9, 10,  9,  9, 10]),
                 'b': array([ 0., nan, nan,  0.,  1.]),
                 'c': array([None, 'e', 'e', 'a', 'z'], dtype=object),
                 'd': array([False, False,  True, False,  True]),
                 'e': array([ 0, 20, 30,  4,  4]),
                 'f': array(['a', None, 'ad', None, 'ad'], dtype=object),
                 'g': array([nan, nan, nan, nan, nan]),
                 'h': array([10, 10, 10, 10, 10]),
                 'i': array([4.2, 4.2, 4.2, 4.2, 4.2]),
                 'j': array(['asdf', 'asdf', 'asdf', 'asdf', 'asdf'], dtype=object)}
        df2 = dx.DataFrame(data2)
        assert_frame_equal(df1, df2)

    def test_append_multiple_old_scalar_columns(self):
        df1 = self.df.append({'a': 10, 'b': 4.2, 'c': 'asdf'}, axis='columns')
        data2 = {'a': array([10, 10, 10, 10, 10]),
                 'b': array([4.2, 4.2, 4.2, 4.2, 4.2]),
                 'c': array(['asdf', 'asdf', 'asdf', 'asdf', 'asdf'], dtype=object),
                 'd': array([False, False,  True, False,  True]),
                 'e': array([ 0, 20, 30,  4,  4]),
                 'f': array(['a', None, 'ad', None, 'ad'], dtype=object),
                 'g': array([nan, nan, nan, nan, nan])}
        df2 = dx.DataFrame(data2)
        assert_frame_equal(df1, df2)

    def test_append_mix_old_new_scalar(self):
        df1 = self.df.append({'a': 10, 'b': 4.2, 'h': 'asdf'}, axis='columns')
        data2 = {'a': array([10, 10, 10, 10, 10]),
                 'b': array([4.2, 4.2, 4.2, 4.2, 4.2]),
                 'c': array([None, 'e', 'e', 'a', 'z'], dtype=object),
                 'd': array([False, False,  True, False,  True]),
                 'e': array([ 0, 20, 30,  4,  4]),
                 'f': array(['a', None, 'ad', None, 'ad'], dtype=object),
                 'g': array([nan, nan, nan, nan, nan]),
                 'h': array(['asdf', 'asdf', 'asdf', 'asdf', 'asdf'], dtype=object)}
        df2 = dx.DataFrame(data2)
        assert_frame_equal(df1, df2)

    def test_append_one_array(self):
        df1 = self.df.append({'h': np.arange(5)}, axis='columns')
        data2 = {'a': array([ 9, 10,  9,  9, 10]),
                 'b': array([ 0., nan, nan,  0.,  1.]),
                 'c': array([None, 'e', 'e', 'a', 'z'], dtype=object),
                 'd': array([False, False,  True, False,  True]),
                 'e': array([ 0, 20, 30,  4,  4]),
                 'f': array(['a', None, 'ad', None, 'ad'], dtype=object),
                 'g': array([nan, nan, nan, nan, nan]),
                 'h': array([0, 1, 2, 3, 4])}
        df2 = dx.DataFrame(data2)
        assert_frame_equal(df1, df2)

    def test_append_multiple_arrays(self):
        df1 = self.df.append({'h': np.arange(5), 'i': np.linspace(2.4, 20.9, 5)}, axis='columns')
        data2 = {'a': array([ 9, 10,  9,  9, 10]),
                 'b': array([ 0., nan, nan,  0.,  1.]),
                 'c': array([None, 'e', 'e', 'a', 'z'], dtype=object),
                 'd': array([False, False,  True, False,  True]),
                 'e': array([ 0, 20, 30,  4,  4]),
                 'f': array(['a', None, 'ad', None, 'ad'], dtype=object),
                 'g': array([nan, nan, nan, nan, nan]),
                 'h': array([0, 1, 2, 3, 4]),
                 'i': array([ 2.4  ,  7.025, 11.65 , 16.275, 20.9  ])}
        df2 = dx.DataFrame(data2)
        assert_frame_equal(df1, df2)

    def append_multiple_new_old_arrays(self):
        df1 = self.df.append({'a': 10, 'b': np.arange(10, 15),
                              'h': np.arange(5), 'i': np.linspace(2.4, 20.9, 5)}, axis='columns')
        data2 = {'a': array([10, 10, 10, 10, 10]),
                 'b': array([10, 11, 12, 13, 14]),
                 'c': array([None, 'e', 'e', 'a', 'z'], dtype=object),
                 'd': array([False, False,  True, False,  True]),
                 'e': array([ 0, 20, 30,  4,  4]),
                 'f': array(['a', None, 'ad', None, 'ad'], dtype=object),
                 'g': array([ 2.4  ,  7.025, 11.65 , 16.275, 20.9  ]),
                 'h': array([0, 1, 2, 3, 4]),
                 'i': array([ 2.4  ,  7.025, 11.65 , 16.275, 20.9  ])}
        df2 = dx.DataFrame(data2)
        assert_frame_equal(df1, df2)

    def test_function_scalar(self):
        df1 = self.df.append({'h': lambda df: df[:, 'e'].max()}, axis='columns')
        data2 = {'a': array([ 9, 10,  9,  9, 10]),
                 'b': array([ 0., nan, nan,  0.,  1.]),
                 'c': array([None, 'e', 'e', 'a', 'z'], dtype=object),
                 'd': array([False, False,  True, False,  True]),
                 'e': array([ 0, 20, 30,  4,  4]),
                 'f': array(['a', None, 'ad', None, 'ad'], dtype=object),
                 'g': array([nan, nan, nan, nan, nan]),
                 'h': array([30, 30, 30, 30, 30])}
        df2 = dx.DataFrame(data2)
        assert_frame_equal(df1, df2)

    def test_function_series(self):
        df1 = self.df.append({'h': lambda df: df[:, 'e'].cumsum() / 3.2}, axis='columns')
        data2 = {'a': array([ 9, 10,  9,  9, 10]),
                 'b': array([ 0., nan, nan,  0.,  1.]),
                 'c': array([None, 'e', 'e', 'a', 'z'], dtype=object),
                 'd': array([False, False,  True, False,  True]),
                 'e': array([ 0, 20, 30,  4,  4]),
                 'f': array(['a', None, 'ad', None, 'ad'], dtype=object),
                 'g': array([nan, nan, nan, nan, nan]),
                 'h': array([ 0.   ,  6.25 , 15.625, 16.875, 18.125])}
        df2 = dx.DataFrame(data2)
        assert_frame_equal(df1, df2)

        df1 = self.df.append({'d': lambda df: df[:, 'e'].cumsum() / 3.2}, axis='columns')
        data2 = {'a': array([ 9, 10,  9,  9, 10]),
                 'b': array([ 0., nan, nan,  0.,  1.]),
                 'c': array([None, 'e', 'e', 'a', 'z'], dtype=object),
                 'd': array([ 0.   ,  6.25 , 15.625, 16.875, 18.125]),
                 'e': array([ 0, 20, 30,  4,  4]),
                 'f': array(['a', None, 'ad', None, 'ad'], dtype=object),
                 'g': array([nan, nan, nan, nan, nan])}
        df2 = dx.DataFrame(data2)
        assert_frame_equal(df1, df2)

        df1 = self.df.append({'d': lambda df: df[:, ['e']].cumsum() / 3.2}, axis='columns')
        data2 = {'a': array([ 9, 10,  9,  9, 10]),
                 'b': array([ 0., nan, nan,  0.,  1.]),
                 'c': array([None, 'e', 'e', 'a', 'z'], dtype=object),
                 'd': array([ 0.   ,  6.25 , 15.625, 16.875, 18.125]),
                 'e': array([ 0, 20, 30,  4,  4]),
                 'f': array(['a', None, 'ad', None, 'ad'], dtype=object),
                 'g': array([nan, nan, nan, nan, nan])}
        df2 = dx.DataFrame(data2)
        assert_frame_equal(df1, df2)

    def test_inputs(self):
        with pytest.raises(TypeError):
            self.df.append(5, axis='columns')

        with pytest.raises(TypeError):
            self.df.append({'h': [1, 2]}, axis='columns')

        with pytest.raises(ValueError):
            self.df.append({'h': lambda x: x.max()}, axis='columns')

    def test_append_one_df(self):
        data0 = {'asdf': [1, 4.4], 'wer': [5, 10], 'c': ['asf', 'ewr'], 'a': [True, False],
                 'wers': [False, True]}
        df0 = dx.DataFrame(data0)
        df1 = self.df.append(df0)
        data2 = {'a': array([9., 10., 9., 9., 10., 1., 4.4]),
                 'b': array([0., nan, nan, 0., 1., 5., 10.]),
                 'c': array([None, 'e', 'e', 'a', 'z', 'asf', 'ewr'], dtype=object),
                 'd': array([False, False, True, False, True, True, False]),
                 'e': array([0, 20, 30, 4, 4, 0, 1]),
                 'f': array(['a', None, 'ad', None, 'ad', None, None], dtype=object),
                 'g': array([nan, nan, nan, nan, nan, nan, nan])}
        df2 = dx.DataFrame(data2)
        assert_frame_equal(df1, df2)

    def test_append_multiple_df_rows(self):
        data0 = {'asdf': [1, 4.4], 'wer': [5, 10], 'c': ['asf', 'ewr'], 'a': [True, False],
                 'wers': [False, True]}
        data1 = {'asdf': [3.1, 4.4], 'wer': [15, 99], 'c': ['TEE', 'ewr']}
        df0 = dx.DataFrame(data0)
        df1 = dx.DataFrame(data1)
        df2 = self.df.append([df0, df1])

        data3 = {'a': array([ 9. , 10. ,  9. ,  9. , 10. ,  1. ,  4.4,  3.1,  4.4]),
                 'b': array([ 0., nan, nan,  0.,  1.,  5., 10., 15., 99.]),
                 'c': array([None, 'e', 'e', 'a', 'z', 'asf', 'ewr', 'TEE', 'ewr'], dtype=object),
                 'd': array([ 0.,  0.,  1.,  0.,  1.,  1.,  0., nan, nan]),
                 'e': array([ 0., 20., 30.,  4.,  4.,  0.,  1., nan, nan]),
                 'f': array(['a', None, 'ad', None, 'ad', None, None, None, None], dtype=object),
                 'g': array([nan, nan, nan, nan, nan, nan, nan, nan, nan])}
        df3 = dx.DataFrame(data3)
        assert_frame_equal(df2, df3)

    def test_append_multiple_df_columns(self):
        data0 = {'asdf': [1, 4.4], 'wer': [5, 10], 'c': ['asf', 'ewr'], 'a': [True, False],
                 'wers': [False, True]}
        data1 = {'asdf': [3.1, 4.4], 'wer': [15, 99], 'c': ['TEE', 'ewr']}
        df0 = dx.DataFrame(data0)
        df1 = dx.DataFrame(data1)
        df2 = self.df.append([df0, df1], axis='columns')
        data3 = {'a': array([ 9, 10,  9,  9, 10]),
                 'b': array([ 0., nan, nan,  0.,  1.]),
                 'c': array([None, 'e', 'e', 'a', 'z'], dtype=object),
                 'd': array([False, False,  True, False,  True]),
                 'e': array([ 0, 20, 30,  4,  4]),
                 'f': array(['a', None, 'ad', None, 'ad'], dtype=object),
                 'g': array([nan, nan, nan, nan, nan]),
                 'asdf': array([1. , 4.4, nan, nan, nan]),
                 'wer': array([ 5., 10., nan, nan, nan]),
                 'c_1': array(['asf', 'ewr', None, None, None], dtype=object),
                 'a_1': array([ 1.,  0., nan, nan, nan]),
                 'wers': array([ 0.,  1., nan, nan, nan]),
                 'asdf_1': array([3.1, 4.4, nan, nan, nan]),
                 'wer_1': array([15., 99., nan, nan, nan]),
                 'c_2': array(['TEE', 'ewr', None, None, None], dtype=object)}
        df3 = dx.DataFrame(data3)
        assert_frame_equal(df2, df3)


class TestMelt:

    def test_melt_one(self):
        data = {'state': ['TX', 'CA', 'OK'],
                'orange': [10, 5, 4],
                'apple': [32, 15, 9],
                'watermelons': [18, 4, 12]}
        df = dx.DataFrame(data)
        df1 = df.melt(id_vars='state')
        data1 = {'state': array(['TX', 'CA', 'OK', 'TX', 'CA', 'OK', 'TX', 'CA', 'OK'], dtype=object),
                 'variable': array(['orange', 'orange', 'orange', 'apple', 'apple', 'apple',
                                    'watermelons', 'watermelons', 'watermelons'], dtype=object),
                 'value': array([10,  5,  4, 32, 15,  9, 18,  4, 12])}
        df2 = dx.DataFrame(data1)
        assert_frame_equal(df1, df2)

        df1 = df.melt(id_vars='state', value_vars='orange')
        data1 = {'state': array(['TX', 'CA', 'OK'], dtype=object),
                 'variable': array(['orange', 'orange', 'orange'], dtype=object),
                 'value': array([10,  5,  4])}
        df2 = dx.DataFrame(data1)
        assert_frame_equal(df1, df2)

        df1 = df.melt(id_vars='state', value_vars=['watermelons', 'orange'])
        data1 = {'state': array(['TX', 'CA', 'OK', 'TX', 'CA', 'OK'], dtype=object),
                 'variable': array(['watermelons', 'watermelons', 'watermelons', 'orange', 'orange',
                                    'orange'], dtype=object),
                 'value': array([18,  4, 12, 10,  5,  4])}
        df2 = dx.DataFrame(data1)
        assert_frame_equal(df1, df2)

        df1 = df.melt(id_vars='state', value_vars=['watermelons', 'orange', 'apple'])
        data1 = {'state': array(['TX', 'CA', 'OK', 'TX', 'CA', 'OK', 'TX', 'CA', 'OK'], dtype=object),
                 'variable': array(['watermelons', 'watermelons', 'watermelons', 'orange', 'orange',
                                    'orange', 'apple', 'apple', 'apple'], dtype=object),
                 'value': array([18,  4, 12, 10,  5,  4, 32, 15,  9])}
        df2 = dx.DataFrame(data1)
        assert_frame_equal(df1, df2)

    def test_melt_two(self):
        data = {'state': ['TX', 'CA', 'OK'],
                'orange': [10, 5, 4],
                'apple': [32, 15, 9],
                'watermelons': [18, 4, 12],
                'male': [100, 200, 300],
                'female': [110, 190, 290]}
        df = dx.DataFrame(data)
        df1 = df.melt(id_vars='state', value_vars=[['orange', 'apple'], ['male']])
        data2 = {'state': array(['TX', 'CA', 'OK', 'TX', 'CA', 'OK'], dtype=object),
                 'variable_0': array(['orange', 'orange', 'orange', 'apple', 'apple', 'apple'],
                                     dtype=object),
                 'value_0': array([10,  5,  4, 32, 15,  9]),
                 'variable_1': array(['male', 'male', 'male', None, None, None], dtype=object),
                 'value_1': array([100., 200., 300.,  nan,  nan,  nan])}
        df2 = dx.DataFrame(data2)
        assert_frame_equal(df1, df2)

        df1 = df.melt(id_vars='state', value_vars=[['orange', 'apple'], ['male', 'female']])
        data2 = {'state': array(['TX', 'CA', 'OK', 'TX', 'CA', 'OK'], dtype=object),
                 'variable_0': array(['orange', 'orange', 'orange', 'apple', 'apple', 'apple'],
                                     dtype=object),
                 'value_0': array([10,  5,  4, 32, 15,  9]),
                 'variable_1': array(['male', 'male', 'male', 'female', 'female', 'female'],
                                     dtype=object),
                 'value_1': array([100, 200, 300, 110, 190, 290])}
        df2 = dx.DataFrame(data2)
        assert_frame_equal(df1, df2)

        df1 = df.melt(id_vars='state', value_vars=[['orange', 'apple'], ['male', 'female']],
                      var_name=['fruit', 'sex'], value_name=['pounds', 'count'])
        data2 = {'state': array(['TX', 'CA', 'OK', 'TX', 'CA', 'OK'], dtype=object),
                 'fruit': array(['orange', 'orange', 'orange', 'apple', 'apple', 'apple'],
                                dtype=object),
                 'pounds': array([10,  5,  4, 32, 15,  9]),
                 'sex': array(['male', 'male', 'male', 'female', 'female', 'female'], dtype=object),
                 'count': array([100, 200, 300, 110, 190, 290])}
        df2 = dx.DataFrame(data2)
        assert_frame_equal(df1, df2)


class TestPivot:

    def test_raw_pivot(self):
        data = {'state': ['TX', 'TX', 'TX', 'OK', 'OK', 'OK'],
                'fruit': ['orange', 'apple', 'banana'] * 2,
                'apple': [32, 15, 9, 4.3, 20, 20]}
        df = dx.DataFrame(data)
        df1 = df.pivot('state', 'fruit', 'apple')
        data2 = {'state': array(['OK', 'TX'], dtype=object),
                 'apple': array([20., 15]),
                 'banana': array([20., 9]),
                 'orange': array([4.3, 32])}
        df2 = dx.DataFrame(data2)
        assert_frame_equal(df1, df2)
