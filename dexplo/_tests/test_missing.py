import dexplo as de
import numpy as np
from numpy import nan
import pytest
from dexplo.testing import assert_frame_equal


class TestIsDropNa(object):

    def test_isna(self):
        df = de.DataFrame({'a': [0, 0, 5],
                           'b': [0, 1.5, nan],
                           'c': [nan, 'g', 'b'],
                           'd': [False, False, True],
                           'e': [90, 20, 30],
                           'f': [nan, 10, 4],
                           'g': ['', None, 'ad'],
                           'h': [nan] * 3})
        df1 = df.isna()
        df2 = de.DataFrame({'a': [False, False, False],
                            'b': [False, False, True],
                            'c': [True, False, False],
                            'd': [False, False, False],
                            'e': [False, False, False],
                            'f': [True, False, False],
                            'g': [False, True, False],
                            'h': [True, True, True]})
        assert_frame_equal(df1, df2)

    def test_dropna(self):
        df = de.DataFrame({'a': [0, 0, 5],
                           'b': [0, 1.5, nan],
                           'c': [nan, 'g', 'b'],
                           'd': [False, False, True],
                           'e': [90, 20, 30],
                           'f': [nan, 10, 4],
                           'g': ['', None, 'ad'],
                           'h': [nan] * 3})
        df1 = df.dropna()
        df2 = de.DataFrame({'a': [],
                            'b': [],
                            'c': [],
                            'd': [],
                            'e': [],
                            'f': [],
                            'g': [],
                            'h': []})
        df2 = df2.astype({'a': 'int', 'c': 'str', 'd': 'bool', 'e': 'int', 'g': 'str'})
        assert_frame_equal(df1, df2)

        df1 = df.dropna('columns')
        df2 = de.DataFrame({'a': [0, 0, 5],
                            'd': [False, False, True],
                            'e': [90, 20, 30]})
        assert_frame_equal(df1, df2)

        df1 = df.dropna(how='all')
        df2 = df.copy()
        assert_frame_equal(df1, df2)

        df1 = df.dropna('columns', how='all')
        df2 = de.DataFrame({'a': [0, 0, 5],
                            'b': [0, 1.5, nan],
                            'c': [nan, 'g', 'b'],
                            'd': [False, False, True],
                            'e': [90, 20, 30],
                            'f': [nan, 10, 4],
                            'g': ['', None, 'ad']})
        assert_frame_equal(df1, df2)

        df = de.DataFrame({'a': [0, 0, 5, 1],
                           'b': [0, 1.5, nan, nan],
                           'c': [nan, 'g', 'b', 'asdf'],
                           'd': [False, False, True, True],
                           'e': [90, 20, 30, 1],
                           'f': [nan, 10, 4, nan],
                           'g': ['', None, 'ad', nan],
                           'h': [nan] * 4})

        df1 = df.dropna(thresh=.5)
        df2 = df.copy()
        assert_frame_equal(df1, df2)

        df1 = df.dropna(thresh=.51)
        df2 = df[:3, :]
        assert_frame_equal(df1, df2)

        df1 = df.dropna(thresh=5)
        df2 = df[:3, :]
        assert_frame_equal(df1, df2)

        df1 = df.dropna(thresh=6)
        df2 = df[1:3, :]
        assert_frame_equal(df1, df2)

        df1 = df.dropna('columns', thresh=.75)
        df2 = df[:, ['a', 'c', 'd', 'e']]
        assert_frame_equal(df1, df2)

        df1 = df.dropna(subset=['a', 'd'])
        df2 = df.copy()
        assert_frame_equal(df1, df2)

        df1 = df.dropna(subset=['a', 'd', 'f'])
        df2 = df[[1, 2], :]
        assert_frame_equal(df1, df2)

        df1 = df.dropna(subset=['a', 'd', 'f', 'c'], thresh=.51)
        df2 = df[1:, :]
        assert_frame_equal(df1, df2)

        df1 = df.dropna('columns', subset=[1, 2], thresh=2)
        df2 = df[:, ['a', 'c', 'd', 'e', 'f']]
        assert_frame_equal(df1, df2)


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