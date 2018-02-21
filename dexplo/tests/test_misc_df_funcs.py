import dexplo as de
import numpy as np
from numpy import nan
import pytest
from dexplo.testing import assert_frame_equal


class TestFillNA(object):

    def test_fillna(self):
        data = {'a': [4, nan, nan, nan, 3, 2],
                'b': [None, 'a', 'd', None, None, 'er'],
                'c': [nan, nan, 5, nan, 7, nan]}
        df = de.DataFrame(data)
        df1 = df.fillna(5)
        data = {'a': [4.0, 5.0, 5.0, 5.0, 3.0, 2.0],
                'b': [None, 'a', 'd', None, None, 'er'],
                'c': [5.0, 5.0, 5.0, 5.0, 7.0, 5.0]}
        df2 = de.DataFrame(data)
        assert_frame_equal(df1, df2)

        df1 = df.fillna({'a': 10, 'b': 'poop'})
        data = {'a': [4.0, 10.0, 10.0, 10.0, 3.0, 2.0],
                'b': ['poop', 'a', 'd', 'poop', 'poop', 'er'],
                'c': [nan, nan, 5.0, nan, 7.0, nan]}
        df2 = de.DataFrame(data)
        assert_frame_equal(df1, df2)

        df1 = df.fillna('dupe')
        data = {'a': [4.0, nan, nan, nan, 3.0, 2.0],
                'b': ['dupe', 'a', 'd', 'dupe', 'dupe', 'er'],
                'c': [nan, nan, 5.0, nan, 7.0, nan]}
        df2 = de.DataFrame(data)
        assert_frame_equal(df1, df2)

    def test_bfillna(self):
        data = {'a': [4, nan, nan, nan, 3, 2],
                'b': [None, 'a', 'd', None, None, 'er'],
                'c': [nan, nan, 5, nan, 7, nan]}
        df = de.DataFrame(data)
        df1 = df.fillna(method='bfill')
        df2 = de.DataFrame({'a': [4.0, 3.0, 3.0, 3.0, 3.0, 2.0],
                            'b': ['a', 'a', 'd', 'er', 'er', 'er'],
                            'c': [5.0, 5.0, 5.0, 7.0, 7.0, nan]})
        assert_frame_equal(df1, df2)

        df1 = df.fillna(method='bfill', limit=1)
        df2 = de.DataFrame({'a': [4.0, nan, nan, 3.0, 3.0, 2.0],
                            'b': ['a', 'a', 'd', None, 'er', 'er'],
                            'c': [nan, 5.0, 5.0, 7.0, 7.0, nan]})
        assert_frame_equal(df1, df2)

        with pytest.raises(ValueError):
            df.fillna(method='bfill', limit=0)

        with pytest.raises(ValueError):
            df.fillna(method='bfill', limit=1, fill_function='mean')

        with pytest.raises(ValueError):
            df.fillna(values=10, method='bfill')

    def test_ffill(self):
        data = {'a': [4, nan, nan, nan, 3, 2],
                'b': [None, 'a', 'd', None, None, 'er'],
                'c': [nan, nan, 5, nan, 7, nan]}
        df = de.DataFrame(data)
        df1 = df.fillna(method='ffill')
        df2 = de.DataFrame({'a': [4.0, 4.0, 4.0, 4.0, 3.0, 2.0],
                            'b': [None, 'a', 'd', 'd', 'd', 'er'],
                            'c': [nan, nan, 5.0, 5.0, 7.0, 7.0]})
        assert_frame_equal(df1, df2)

        df1 = df.fillna(method='ffill', limit=1)
        df2 = de.DataFrame({'a': [4.0, 4.0, nan, nan, 3.0, 2.0],
                            'b': [None, 'a', 'd', 'd', None, 'er'],
                            'c': [nan, nan, 5.0, 5.0, 7.0, 7.0]})
        assert_frame_equal(df1, df2)

    def test_fillna_fill_function(self):
        data = {'a': [4, nan, nan, nan, 3, 2],
                'b': [None, 'a', 'd', None, None, 'er'],
                'c': [nan, nan, 5, nan, 7, nan]}
        df = de.DataFrame(data)
        df1 = df.fillna(fill_function='mean')
        df2 = de.DataFrame({'a': [4.0, 3.0, 3.0, 3.0, 3.0, 2.0],
                            'b': [None, 'a', 'd', None, None, 'er'],
                            'c': [6.0, 6.0, 5.0, 6.0, 7.0, 6.0]})
        assert_frame_equal(df1, df2)

        df1 = df.fillna(fill_function='median')
        assert_frame_equal(df1, df2)


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
