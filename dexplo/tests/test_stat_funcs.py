import dexplo as de
import numpy as np
from numpy import nan
import pytest
from dexplo.testing import assert_frame_equal, assert_array_equal, assert_dict_list

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
    df = de.DataFrame({'a': [0, 0, 5],
                       'b': [0, 1.5, np.nan],
                       'c': [''] + list('bg'),
                       'd': [False, False, True],
                       'e': [0, 20, 30],
                       'f': ['', None, 'ad'],
                       'g': np.zeros(3, dtype='int'),
                       'h': [np.nan] * 3})

    def test_sum(self):
        df1 = self.df.sum()
        df2 = de.DataFrame({'a': [5],
                            'b': [1.5],
                            'c': ['bg'],
                            'd': [1],
                            'e': [50],
                            'f': ['ad'],
                            'g': [0],
                            'h': [0.]})
        assert_frame_equal(df1, df2)

        df1 = self.df.sum(axis='columns')
        df2 = de.DataFrame({'sum': [0, 21.5, 36]})
        assert_frame_equal(df1, df2)

        df1 = self.df.select_dtypes('str').sum(axis='columns')
        df2 = de.DataFrame({'sum': ['', 'b', 'gad']})
        assert_frame_equal(df1, df2)

    def test_max(self):
        df1 = self.df.max()
        df2 = de.DataFrame({'a': [5],
                            'b': [1.5],
                            'c': ['g'],
                            'd': [1],
                            'e': [30],
                            'f': ['ad'],
                            'g': [0],
                            'h': [np.nan]})
        assert_frame_equal(df1, df2)

        df1 = self.df.max(axis='columns')
        df2 = de.DataFrame({'max': [0., 20, 30]})
        assert_frame_equal(df1, df2)

        df1 = self.df.select_dtypes('str').max(axis='columns')
        df2 = de.DataFrame({'max': ['', 'b', 'g']})
        assert_frame_equal(df1, df2)

    def test_min(self):
        df1 = self.df.min()
        df2 = de.DataFrame({'a': [0],
                            'b': [0.],
                            'c': [''],
                            'd': [0],
                            'e': [0],
                            'f': [''],
                            'g': [0],
                            'h': [np.nan]})
        assert_frame_equal(df1, df2)

        df1 = self.df.min(axis='columns')
        df2 = de.DataFrame({'min': [0., 0, 0]})
        assert_frame_equal(df1, df2)

        df1 = self.df.select_dtypes('str').min(axis='columns')
        df2 = de.DataFrame({'min': ['', 'b', 'ad']})
        assert_frame_equal(df1, df2)

    def test_mean(self):
        df = de.DataFrame({'a': [0, 5, 16],
                           'b': [4.5, 1.5, np.nan],
                           'c': [''] + list('bg'),
                           'd': [False, False, True],
                           'e': [0, 20, 40],
                           'f': ['', None, 'ad'],
                           'g': np.zeros(3, dtype='int'),
                           'h': [np.nan] * 3})
        df1 = df.mean()
        df2 = de.DataFrame({'a': [7.],
                            'b': [3.],
                            'd': [1 / 3],
                            'e': [20.],
                            'g': [0.],
                            'h': [np.nan]})
        assert_frame_equal(df1, df2)

        df1 = df.mean('columns')
        df2 = de.DataFrame({'mean': [.9, 5.3, 14.25]})
        assert_frame_equal(df1, df2)

        with pytest.raises(TypeError):
            df.select_dtypes('str').mean()

        with pytest.raises(TypeError):
            df.select_dtypes('str').mean('columns')

    def test_median(self):
        df = de.DataFrame({'a': [0, 5, 16],
                           'b': [4.5, 1.5, np.nan],
                           'c': [''] + list('bg'),
                           'd': [False, False, True],
                           'e': [0, 20, 40],
                           'f': ['', None, 'ad'],
                           'g': np.zeros(3, dtype='int'),
                           'h': [np.nan] * 3})
        df1 = df.median()
        df2 = de.DataFrame({'a': [5.],
                            'b': [3.],
                            'd': [0.],
                            'e': [20.],
                            'g': [0.],
                            'h': [np.nan]})
        assert_frame_equal(df1, df2)

        df1 = df.median('columns')
        df2 = de.DataFrame({'median': [0., 1.5, 8.5]})
        assert_frame_equal(df1, df2)

        with pytest.raises(TypeError):
            df.select_dtypes('str').median()

        with pytest.raises(TypeError):
            df.select_dtypes('str').median('columns')

    def test_std(self):
        df = de.DataFrame({'a': [0, 5, 16],
                           'b': [4.5, 1.5, np.nan],
                           'c': [''] + list('bg'),
                           'd': [False, False, True],
                           'e': [0, 20, 40],
                           'f': ['', None, 'ad'],
                           'g': np.zeros(3, dtype='int'),
                           'h': [np.nan] * 3})
        df1 = df.std()
        df2 = de.DataFrame(np.array([[8.18535277, 2.12132034, 0.57735027, 20., 0., nan]]),
                           columns=list('abdegh'))
        assert_frame_equal(df1, df2)

        df1 = df.std('columns')
        df2 = de.DataFrame(np.array([[2.01246118],
                                     [8.46758525],
                                     [18.66145761]]), columns=['std'])
        assert_frame_equal(df1, df2)

        df1 = df.std(ddof=2)
        df2 = de.DataFrame(np.array([[11.5758369, nan, 0.81649658, 28.28427125, 0., nan]]),
                           columns=list('abdegh'))
        assert_frame_equal(df1, df2)

        with pytest.raises(TypeError):
            df.select_dtypes('str').std()

        with pytest.raises(TypeError):
            df.select_dtypes('str').std('columns')

    def test_var(self):
        df = de.DataFrame({'a': [0, 5, 16],
                           'b': [4.5, 1.5, np.nan],
                           'c': [''] + list('bg'),
                           'd': [False, False, True],
                           'e': [0, 20, 40],
                           'f': ['', None, 'ad'],
                           'g': np.zeros(3, dtype='int'),
                           'h': [np.nan] * 3})
        df1 = df.var()
        df2 = de.DataFrame(np.array([[67., 4.5, 1 / 3,
                                      400, 0, nan]]),
                           columns=list('abdegh'))
        assert_frame_equal(df1, df2)

        df1 = df.var('columns')
        df2 = de.DataFrame(np.array([[4.05],
                                     [71.7],
                                     [348.25]]), columns=['var'])
        assert_frame_equal(df1, df2)

        df1 = df.var(ddof=2)
        df2 = de.DataFrame(np.array([[134, nan, 2 / 3, 800., 0., nan]]),
                           columns=list('abdegh'))
        assert_frame_equal(df1, df2)

        with pytest.raises(TypeError):
            df.select_dtypes('str').var()

        with pytest.raises(TypeError):
            df.select_dtypes('str').var('columns')

    def test_abs(self):
        df = de.DataFrame({'a': [0, -5, -16],
                           'b': [4.5, -1.5, np.nan],
                           'c': [''] + list('bg'),
                           'd': [False, False, True],
                           'e': [0, -20, 40],
                           'f': ['', None, 'ad'],
                           'g': np.zeros(3, dtype='int'),
                           'h': [np.nan] * 3})

        df1 = df.abs(True)
        df2 = de.DataFrame({'a': [0, 5, 16],
                            'b': [4.5, 1.5, np.nan],
                            'c': [''] + list('bg'),
                            'd': [False, False, True],
                            'e': [0, 20, 40],
                            'f': ['', None, 'ad'],
                            'g': np.zeros(3, dtype='int'),
                            'h': [np.nan] * 3})
        assert_frame_equal(df1, df2)

        df1 = df.abs()
        df2 = de.DataFrame({'a': [0, 5, 16],
                            'b': [4.5, 1.5, np.nan],
                            'd': [False, False, True],
                            'e': [0, 20, 40],
                            'g': np.zeros(3, dtype='int'),
                            'h': [np.nan] * 3})
        assert_frame_equal(df1, df2)

        df1 = abs(df)
        df2 = de.DataFrame({'a': [0, 5, 16],
                            'b': [4.5, 1.5, np.nan],
                            'd': [False, False, True],
                            'e': [0, 20, 40],
                            'g': np.zeros(3, dtype='int'),
                            'h': [np.nan] * 3})
        assert_frame_equal(df1, df2)

    def test_any(self):
        df = de.DataFrame({'a': [0, -5, -16],
                           'b': [0, -1.5, np.nan],
                           'c': [''] + list('bg'),
                           'd': [False, False, True],
                           'e': [0, -20, 40],
                           'f': ['', None, 'ad'],
                           'g': np.zeros(3, dtype='int'),
                           'h': [np.nan] * 3})
        df1 = df.any()
        df2 = de.DataFrame({'a': [True],
                            'b': [True],
                            'c': [True],
                            'd': [True],
                            'e': [True],
                            'f': [True],
                            'g': [False],
                            'h': [False]})
        assert_frame_equal(df1, df2)

        df1 = df.any('columns')
        df2 = de.DataFrame({'any': [False, True, True]})
        assert_frame_equal(df1, df2)

    def test_all(self):
        df = de.DataFrame({'a': [1, -5, -16],
                           'b': [0, -1.5, np.nan],
                           'c': [''] + list('bg'),
                           'd': [False, False, True],
                           'e': [0, -20, 40],
                           'f': ['', None, 'ad'],
                           'g': np.zeros(3, dtype='int'),
                           'h': [np.nan] * 3})
        df1 = df.all()
        df2 = de.DataFrame({'a': [True],
                            'b': [False],
                            'c': [False],
                            'd': [False],
                            'e': [False],
                            'f': [False],
                            'g': [False],
                            'h': [False]})
        assert_frame_equal(df1, df2)

        df1 = df.all('columns')
        df2 = de.DataFrame({'all': [False, False, False]})
        assert_frame_equal(df1, df2)

    def test_argmax(self):
        df = de.DataFrame({'a': [0, 0, 5],
                           'b': [0, 1.5, np.nan],
                           'c': [''] + list('bb'),
                           'd': [False, False, True],
                           'e': [90, 20, 30],
                           'f': ['', None, 'ad'],
                           'g': np.zeros(3, dtype='int'),
                           'h': [np.nan] * 3})
        df1 = df.argmax()
        df2 = de.DataFrame({'a': [2],
                            'b': [1.],
                            'c': [1.],
                            'd': [2],
                            'e': [0],
                            'f': [2.],
                            'g': [0],
                            'h': [nan]},
                           columns=list('abcdefgh'))
        assert_frame_equal(df1, df2)

        df1 = df.argmax('columns')
        df2 = de.DataFrame({'argmax': [3., 3, 3]})
        assert_frame_equal(df1, df2)

    def test_argmin(self):
        df = de.DataFrame({'a': [0, 0, 5],
                           'b': [0, 1.5, np.nan],
                           'c': [''] + list('bb'),
                           'd': [False, False, True],
                           'e': [90, 20, 30],
                           'f': ['', None, 'ad'],
                           'g': np.zeros(3, dtype='int'),
                           'h': [np.nan] * 3})
        df1 = df.argmin()
        df2 = de.DataFrame({'a': [0],
                            'b': [0.],
                            'c': [0.],
                            'd': [0],
                            'e': [1],
                            'f': [0.],
                            'g': [0],
                            'h': [nan]},
                           columns=list('abcdefgh'))
        assert_frame_equal(df1, df2)

        df1 = df.argmin('columns')
        df2 = de.DataFrame({'argmin': [0., 0, 4]})
        assert_frame_equal(df1, df2)

    def test_count(self):
        df = de.DataFrame({'a': [0, 0, 5],
                           'b': [0, 1.5, np.nan],
                           'c': [''] + list('bb'),
                           'd': [False, False, True],
                           'e': [90, 20, 30],
                           'f': ['', None, 'ad'],
                           'g': np.zeros(3, dtype='int'),
                           'h': [np.nan] * 3})
        df1 = df.count()
        df2 = de.DataFrame({'a': [3],
                            'b': [2],
                            'c': [3],
                            'd': [3],
                            'e': [3],
                            'f': [2],
                            'g': [3],
                            'h': [0]},
                           columns=list('abcdefgh'))
        assert_frame_equal(df1, df2)

        df1 = df.count('columns')
        df2 = de.DataFrame({'count': [7, 6, 6]})
        assert_frame_equal(df1, df2)

    def test_clip(self):
        df = de.DataFrame({'a': [0, 0, 5],
                           'b': [0, 1.5, np.nan],
                           'c': [''] + list('bb'),
                           'd': [False, False, True],
                           'e': [90, 20, 30],
                           'f': ['', None, 'ad'],
                           'g': np.zeros(3, dtype='int'),
                           'h': [np.nan] * 3})

        df1 = df.clip(2)
        df2 = de.DataFrame({'a': [2, 2, 5],
                            'b': [2, 2, np.nan],
                            'd': [2, 2, 2],
                            'e': [90, 20, 30],
                            'g': [2, 2, 2],
                            'h': [np.nan] * 3})
        assert_frame_equal(df1, df2)

        df1 = df.clip(2, keep=True)
        df2 = de.DataFrame({'a': [2, 2, 5],
                            'b': [2, 2, np.nan],
                            'c': [''] + list('bb'),
                            'd': [2, 2, 2],
                            'e': [90, 20, 30],
                            'f': ['', None, 'ad'],
                            'g': [2, 2, 2],
                            'h': [np.nan] * 3})
        assert_frame_equal(df1, df2)

        df1 = df.clip(upper=10)
        df2 = de.DataFrame({'a': [0, 0, 5],
                            'b': [0, 1.5, np.nan],
                            'd': [0, 0, 1],
                            'e': [10, 10, 10],
                            'g': np.zeros(3, dtype='int'),
                            'h': [np.nan] * 3})
        assert_frame_equal(df1, df2)

        df1 = df.clip(lower=1, upper=2)
        df2 = de.DataFrame({'a': [1, 1, 2],
                            'b': [1, 1.5, np.nan],
                            'd': [1, 1, 1],
                            'e': [2, 2, 2],
                            'g': [1, 1, 1],
                            'h': [np.nan] * 3})
        assert_frame_equal(df1, df2)

        df1 = df.clip(lower=1, upper=2, keep=True)
        df2 = de.DataFrame({'a': [1, 1, 2],
                            'b': [1, 1.5, np.nan],
                            'c': [''] + list('bb'),
                            'd': [1, 1, 1],
                            'e': [2, 2, 2],
                            'f': ['', None, 'ad'],
                            'g': [1, 1, 1],
                            'h': [np.nan] * 3})
        assert_frame_equal(df1, df2)

        df1 = df.clip('a')
        df2 = de.DataFrame({'c': ['a', 'b', 'b'],
                            'f': ['a', None, 'ad']})
        assert_frame_equal(df1, df2)

        df1 = df.clip(lower='a', upper='ab')
        df2 = de.DataFrame({'c': ['a', 'ab', 'ab'],
                            'f': ['a', None, 'ab']})
        assert_frame_equal(df1, df2)

        df1 = df.clip(lower='a', upper='ab', keep=True)
        df2 = de.DataFrame({'a': [0, 0, 5],
                            'b': [0, 1.5, np.nan],
                            'c': ['a', 'ab', 'ab'],
                            'd': [False, False, True],
                            'e': [90, 20, 30],
                            'f': ['a', None, 'ab'],
                            'g': np.zeros(3, dtype='int'),
                            'h': [np.nan] * 3})
        assert_frame_equal(df1, df2)

        with pytest.raises(ValueError):
            df.clip(10, 2)

        with pytest.raises(ValueError):
            df.clip('Z', 'ER')

    def test_cummax(self):
        df = de.DataFrame({'a': [0, 0, 5],
                           'b': [0, 1.5, nan],
                           'c': ['', 'g', 'b'],
                           'd': [False, False, True],
                           'e': [90, 20, 30],
                           'f': [nan, 10, 4],
                           'g': ['', None, 'ad'],
                           'h': [nan] * 3})
        df1 = df.cummax()
        df2 = de.DataFrame({'a': [0, 0, 5],
                            'b': [0, 1.5, 1.5],
                            'c': ['', 'g', 'g'],
                            'd': [False, False, True],
                            'e': [90, 90, 90],
                            'f': [nan, 10, 10],
                            'g': ['', '', 'ad'],
                            'h': [np.nan] * 3})
        assert_frame_equal(df1, df2)

        df1 = df.cummax('columns')
        data = np.array([[0., 0., 0., 90., 90., 90.],
                         [0., 1.5, 1.5, 20., 20., 20.],
                         [5., 5., 5., 30., 30., 30.]])
        df2 = de.DataFrame(data, columns=list('abdefh'))
        assert_frame_equal(df1, df2)

        df1 = df.select_dtypes('str').cummax('columns')
        df2 = de.DataFrame({'c': ['', 'g', 'b'],
                            'g': ['', 'g', 'b']})
        assert_frame_equal(df1, df2)

    def test_cummin(self):
        df = de.DataFrame({'a': [0, 0, 5],
                           'b': [0, 1.5, nan],
                           'c': ['', 'g', 'b'],
                           'd': [False, False, True],
                           'e': [90, 20, 30],
                           'f': [nan, 10, 4],
                           'g': ['', None, 'ad'],
                           'h': [nan] * 3})
        df1 = df.cummin()
        df2 = de.DataFrame({'a': [0, 0, 0],
                            'b': [0., 0, 0],
                            'c': ['', '', ''],
                            'd': [False, False, False],
                            'e': [90, 20, 20],
                            'f': [nan, 10, 4],
                            'g': ['', '', ''],
                            'h': [np.nan] * 3})
        assert_frame_equal(df1, df2)

        df1 = df.cummin('columns')
        data = np.array([[0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0.],
                         [5., 5., 1., 1., 1., 1.]])
        df2 = de.DataFrame(data, columns=list('abdefh'))
        assert_frame_equal(df1, df2)

        df1 = df.select_dtypes('str').cummin('columns')
        df2 = de.DataFrame({'c': ['', 'g', 'b'],
                            'g': ['', 'g', 'ad']})
        assert_frame_equal(df1, df2)

    def test_cumsum(self):
        df = de.DataFrame({'a': [0, 0, 5],
                           'b': [0, 1.5, nan],
                           'c': ['', 'g', 'b'],
                           'd': [False, False, True],
                           'e': [90, 20, 30],
                           'f': [nan, 10, 4],
                           'g': ['', None, 'ad'],
                           'h': [nan] * 3})
        df1 = df.cumsum()
        df2 = de.DataFrame({'a': [0, 0, 5],
                            'b': [0, 1.5, 1.5],
                            'c': ['', 'g', 'gb'],
                            'd': [0, 0, 1],
                            'e': [90, 110, 140],
                            'f': [0., 10, 14],
                            'g': ['', '', 'ad'],
                            'h': [0., 0, 0]})
        assert_frame_equal(df1, df2)

        df1 = df.cumsum('columns')
        data = np.array([[0., 0., 0., 90., 90., 90.],
                         [0., 1.5, 1.5, 21.5, 31.5, 31.5],
                         [5., 5., 6., 36., 40., 40.]])
        df2 = de.DataFrame(data, columns=list('abdefh'))
        assert_frame_equal(df1, df2)

        df1 = df.select_dtypes('str').cumsum('columns')
        df2 = de.DataFrame({'c': ['', 'g', 'b'],
                            'g': ['', 'g', 'bad']})
        assert_frame_equal(df1, df2)

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
