import dexplo as de
import numpy as np
from numpy import nan
import pytest
from dexplo.testing import assert_frame_equal


class TestSimpleAggs(object):
    df = de.DataFrame({'a': [0, 0, 5],
                       'b': [0, 1.5, np.nan],
                       'c': [''] + list('bg'),
                       'd': [False, False, True],
                       'e': [0, 20, 30],
                       'f': ['', None, 'ad'],
                       'g': np.zeros(3, dtype='int64'),
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
                           'g': np.zeros(3, dtype='int64'),
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
                           'g': np.zeros(3, dtype='int64'),
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
                           'g': np.zeros(3, dtype='int64'),
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
                           'g': np.zeros(3, dtype='int64'),
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

    def test_count(self):
        df = de.DataFrame({'a': [0, 0, 5],
                           'b': [0, 1.5, np.nan],
                           'c': [''] + list('bb'),
                           'd': [False, False, True],
                           'e': [90, 20, 30],
                           'f': ['', None, 'ad'],
                           'g': np.zeros(3, dtype='int64'),
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


class TestAnyAll(object):

    def test_any(self):
        df = de.DataFrame({'a': [0, -5, -16],
                           'b': [0, -1.5, np.nan],
                           'c': [''] + list('bg'),
                           'd': [False, False, True],
                           'e': [0, -20, 40],
                           'f': ['', None, 'ad'],
                           'g': np.zeros(3, dtype='int64'),
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
                           'g': np.zeros(3, dtype='int64'),
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


class TestArgMinMax(object):

    def test_argmax(self):
        df = de.DataFrame({'a': [0, 0, 5],
                           'b': [0, 1.5, np.nan],
                           'c': [''] + list('bb'),
                           'd': [False, False, True],
                           'e': [90, 20, 30],
                           'f': ['', None, 'ad'],
                           'g': np.zeros(3, dtype='int64'),
                           'h': [np.nan] * 3})
        df1 = df.argmax()
        df2 = de.DataFrame({'a': [2],
                            'b': [1.],
                            'c': [1],
                            'd': [2],
                            'e': [0],
                            'f': [2],
                            'g': [0],
                            'h': [nan]},
                           columns=list('abcdefgh'))
        assert_frame_equal(df1, df2)

        df1 = df.argmax('columns')
        df2 = de.DataFrame({'argmax': [3, 3, 3]})
        assert_frame_equal(df1, df2)

    def test_argmin(self):
        df = de.DataFrame({'a': [0, 0, 5],
                           'b': [0, 1.5, np.nan],
                           'c': [''] + list('bb'),
                           'd': [False, False, True],
                           'e': [90, 20, 30],
                           'f': ['', None, 'ad'],
                           'g': np.zeros(3, dtype='int64'),
                           'h': [np.nan] * 3})
        df1 = df.argmin()
        df2 = de.DataFrame({'a': [0],
                            'b': [0.],
                            'c': [0],
                            'd': [0],
                            'e': [1],
                            'f': [0],
                            'g': [0],
                            'h': [nan]},
                           columns=list('abcdefgh'))
        assert_frame_equal(df1, df2)

        df1 = df.argmin('columns')
        df2 = de.DataFrame({'argmin': [0, 0, 4]})
        assert_frame_equal(df1, df2)


class TestAbsClip(object):

    def test_abs(self):
        df = de.DataFrame({'a': [0, -5, -16],
                           'b': [4.5, -1.5, np.nan],
                           'c': [''] + list('bg'),
                           'd': [False, False, True],
                           'e': [0, -20, 40],
                           'f': ['', None, 'ad'],
                           'g': np.zeros(3, dtype='int64'),
                           'h': [np.nan] * 3})

        df1 = df.abs(True)
        df2 = de.DataFrame({'a': [0, 5, 16],
                            'b': [4.5, 1.5, np.nan],
                            'c': [''] + list('bg'),
                            'd': [False, False, True],
                            'e': [0, 20, 40],
                            'f': ['', None, 'ad'],
                            'g': np.zeros(3, dtype='int64'),
                            'h': [np.nan] * 3})
        assert_frame_equal(df1, df2)

        df1 = df.abs()
        df2 = de.DataFrame({'a': [0, 5, 16],
                            'b': [4.5, 1.5, np.nan],
                            'd': [False, False, True],
                            'e': [0, 20, 40],
                            'g': np.zeros(3, dtype='int64'),
                            'h': [np.nan] * 3})
        assert_frame_equal(df1, df2)

        df1 = abs(df)
        df2 = de.DataFrame({'a': [0, 5, 16],
                            'b': [4.5, 1.5, np.nan],
                            'd': [False, False, True],
                            'e': [0, 20, 40],
                            'g': np.zeros(3, dtype='int64'),
                            'h': [np.nan] * 3})
        assert_frame_equal(df1, df2)

    def test_clip(self):
        df = de.DataFrame({'a': [0, 0, 5],
                           'b': [0, 1.5, np.nan],
                           'c': [''] + list('bb'),
                           'd': [False, False, True],
                           'e': [90, 20, 30],
                           'f': ['', None, 'ad'],
                           'g': np.zeros(3, dtype='int64'),
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
                            'g': np.zeros(3, dtype='int64'),
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
                            'g': np.zeros(3, dtype='int64'),
                            'h': [np.nan] * 3})
        assert_frame_equal(df1, df2)

        with pytest.raises(ValueError):
            df.clip(10, 2)

        with pytest.raises(ValueError):
            df.clip('Z', 'ER')


class TestCum(object):

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


class TestCovCorr(object):

    def test_cov(self):
        df = de.DataFrame({'a': [0, 0, 5, 1],
                           'b': [0, 1.5, nan, nan],
                           'c': [nan, 'g', 'b', 'asdf'],
                           'd': [False, False, True, True],
                           'e': [90, 20, 30, 1],
                           'f': [nan, 10, 4, nan],
                           'g': ['', None, 'ad', nan],
                           'h': [nan] * 4})
        df1 = df.cov()
        data = {'Column Name': ['a', 'b', 'd', 'e', 'f', 'h'],
                'a': [5.666666666666667, 0.0, 1.0, -20.166666666666668, -15.0, nan],
                'b': [0.0, 1.125, 0.0, -52.5, nan, nan],
                'd': [1.0, 0.0, 0.3333333333333333, -13.166666666666666, -3.0, nan],
                'e': [-20.166666666666668,
                      -52.5,
                      -13.166666666666666,
                      1476.9166666666667,
                      -30.0,
                      nan],
                'f': [-15.0, nan, -3.0, -30.0, 18.0, nan],
                'h': [nan, nan, nan, nan, nan, nan]}
        df2 = de.DataFrame(data)
        assert_frame_equal(df1, df2)

        df1 = df.select_dtypes('int').cov()
        data = {'Column Name': ['a', 'e'],
                'a': [5.666666666666667, -20.166666666666668],
                'e': [-20.166666666666668, 1476.9166666666667]}
        df2 = de.DataFrame(data)
        assert_frame_equal(df1, df2)

        with pytest.raises(TypeError):
            df.select_dtypes('str').cov()

    def test_corr(self):
        df = de.DataFrame({'a': [0, 0, 5, 1],
                           'b': [0, 1.5, nan, nan],
                           'c': [nan, 'g', 'b', 'asdf'],
                           'd': [False, False, True, True],
                           'e': [90, 20, 30, 1],
                           'f': [nan, 10, 4, nan],
                           'g': ['', None, 'ad', nan],
                           'h': [nan] * 4})
        df1 = df.corr()
        data = {'Column Name': ['a', 'b', 'd', 'e', 'f', 'h'],
                'a': [1.0, nan, 0.7276068751089989, -0.2204409585872243, -1.0, nan],
                'b': [nan, 1.0000000000000002, nan, -1.0, nan, nan],
                'd': [0.7276068751089989, nan, 1.0, -0.5934149352143404, -1.0, nan],
                'e': [-0.2204409585872243,
                      -1.0,
                      -0.5934149352143404,
                      1.0000000000000002,
                      -1.0,
                      nan],
                'f': [-1.0, nan, -1.0, -1.0, 1.0000000000000002, nan],
                'h': [nan, nan, nan, nan, nan, nan]}
        df2 = de.DataFrame(data)
        assert_frame_equal(df1, df2)

        df1 = df.select_dtypes('int').corr()
        data = {'Column Name': ['a', 'e'],
                'a': [1.0, -0.2204409585872243],
                'e': [-0.2204409585872243, 1.]}
        df2 = de.DataFrame(data)
        assert_frame_equal(df1, df2)

        with pytest.raises(TypeError):
            df.select_dtypes('str').corr()

    def test_cov2(self):
        data = {'a': [0, 0, 5, 9, 3, 4, 5, 1],
                'b': [0, 1.512344353, 8, 9, np.nan, 3, 2, 8],
                'c': [''] + list('bgggzgh'),
                'd': [False, False, True, False] * 2,
                'e': [0, 20, 30, 4, 5, 6, 7, 8],
                'f': [0., 3, 3, 3, 11, 4, 5, 1],
                'g': ['', nan, 'ad', 'effd', 'ef', None, 'ett', 'zzzz'],
                'h': [0, 4, 5, 6, 7, 8, 9, 0],
                'i': np.array([0, 7, 6, 5, 4, 3, 2, 11]),
                'j': np.zeros(8, dtype='int64'),
                'k': np.ones(8) - 1,
                'l': [np.nan] * 8}

        df = de.DataFrame(data)
        df1 = df.cov()
        data = {'Column Name': ['a', 'b', 'd', 'e', 'f', 'h', 'i', 'j', 'k', 'l'],
                'a': [9.410714285714286,
                      7.159612750666668,
                      0.4642857142857143,
                      -0.2857142857142857,
                      2.25,
                      6.482142857142857,
                      -1.1785714285714286,
                      0.0,
                      0.0,
                      nan],
                'b': [7.159612750666668,
                      13.737677416007287,
                      0.1660788403333333,
                      10.76910435583333,
                      0.0005878263333372047,
                      -0.3345089859999983,
                      8.587742030833335,
                      0.0,
                      0.0,
                      nan],
                'd': [0.4642857142857143,
                      0.1660788403333333,
                      0.21428571428571427,
                      2.4285714285714284,
                      0.07142857142857142,
                      0.6071428571428571,
                      -0.21428571428571427,
                      0.0,
                      0.0,
                      nan],
                'e': [-0.2857142857142857,
                      10.76910435583333,
                      2.4285714285714284,
                      98.57142857142857,
                      -2.2857142857142856,
                      1.4285714285714286,
                      14.285714285714286,
                      0.0,
                      0.0,
                      nan],
                'f': [2.25,
                      0.0005878263333372047,
                      0.07142857142857142,
                      -2.2857142857142856,
                      11.071428571428571,
                      7.535714285714286,
                      -1.6428571428571428,
                      0.0,
                      0.0,
                      nan],
                'h': [6.482142857142857,
                      -0.3345089859999983,
                      0.6071428571428571,
                      1.4285714285714286,
                      7.535714285714286,
                      11.553571428571429,
                      -3.892857142857143,
                      0.0,
                      0.0,
                      nan],
                'i': [-1.1785714285714286,
                      8.587742030833335,
                      -0.21428571428571427,
                      14.285714285714286,
                      -1.6428571428571428,
                      -3.892857142857143,
                      11.357142857142858,
                      0.0,
                      0.0,
                      nan],
                'j': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, nan],
                'k': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, nan],
                'l': [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]}
        df2 = de.DataFrame(data, columns=['Column Name', 'a', 'b', 'd', 'e', 'f',
                                          'h', 'i', 'j', 'k', 'l'])
        assert_frame_equal(df1, df2)

        np.random.seed(1)
        a = np.random.randint(0, 100, (10, 5))
        df = de.DataFrame(a)
        df1 = df.cov()
        data = {'Column Name': ['a0', 'a1', 'a2', 'a3', 'a4'],
                'a0': [828.6222222222223,
                       204.88888888888889,
                       -51.422222222222224,
                       246.06666666666666,
                       78.33333333333333],
                'a1': [204.88888888888889,
                       1434.9444444444443,
                       -126.72222222222223,
                       151.38888888888889,
                       -548.8888888888889],
                'a2': [-51.422222222222224,
                       -126.72222222222223,
                       1321.6555555555556,
                       -151.58888888888882,
                       -466.22222222222223],
                'a3': [246.06666666666666,
                       151.38888888888889,
                       -151.58888888888882,
                       779.1222222222223,
                       -5.111111111111111],
                'a4': [78.33333333333333,
                       -548.8888888888889,
                       -466.22222222222223,
                       -5.111111111111111,
                       1007.5555555555555]}
        df2 = de.DataFrame(data)
        assert_frame_equal(df1, df2)

    def test_corr2(self):
        data = {'a': [0, 0, 5, 9, 3, 4, 5, 1],
                'b': [0, 1.512344353, 8, 9, np.nan, 3, 2, 8],
                'c': [''] + list('bgggzgh'),
                'd': [False, False, True, False] * 2,
                'e': [0, 20, 30, 4, 5, 6, 7, 8],
                'f': [0., 3, 3, 3, 11, 4, 5, 1],
                'g': ['', nan, 'ad', 'effd', 'ef', None, 'ett', 'zzzz'],
                'h': [0, 4, 5, 6, 7, 8, 9, 0],
                'i': np.array([0, 7, 6, 5, 4, 3, 2, 11]),
                'j': np.zeros(8, dtype='int64'),
                'k': np.ones(8) - 1,
                'l': [np.nan] * 8}

        df = de.DataFrame(data)
        df1 = df.corr()
        data = {'Column Name': ['a', 'b', 'd', 'e', 'f', 'h', 'i', 'j', 'k', 'l'],
                'a': [1.0,
                      0.583685097392588,
                      0.326947045638107,
                      -0.009380913603,
                      0.22042933617057,
                      0.6216546364862136,
                      -0.1140013699089899,
                      nan,
                      nan,
                      nan],
                'b': [0.583685097392588,
                      1.0,
                      0.09182951524990407,
                      0.2767305320147613,
                      9.30544422486969e-05,
                      -0.02540609057931423,
                      0.6391126877826773,
                      nan,
                      nan,
                      nan],
                'd': [0.326947045638107,
                      0.09182951524990407,
                      1.0,
                      0.5284193913361779,
                      0.046373889576016826,
                      0.38586568070322685,
                      -0.137360563948689,
                      nan,
                      nan,
                      nan],
                'e': [-0.009380913603,
                      0.2767305320147613,
                      0.5284193913361779,
                      1.0,
                      -0.06919020001030568,
                      0.042331953245962436,
                      0.4269646211491787,
                      nan,
                      nan,
                      nan],
                'f': [0.22042933617057,
                      9.30544422486969e-05,
                      0.046373889576016826,
                      -0.06919020001030568,
                      1.0,
                      0.6662917960183002,
                      -0.14650870336708577,
                      nan,
                      nan,
                      nan],
                'h': [0.6216546364862136,
                      -0.02540609057931423,
                      0.38586568070322685,
                      0.042331953245962436,
                      0.6662917960183002,
                      1.0,
                      -0.3398410175630919,
                      nan,
                      nan,
                      nan],
                'i': [-0.1140013699089899,
                      0.6391126877826773,
                      -0.137360563948689,
                      0.4269646211491787,
                      -0.14650870336708577,
                      -0.3398410175630919,
                      1.0,
                      nan,
                      nan,
                      nan],
                'j': [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
                'k': [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
                'l': [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]}
        df2 = de.DataFrame(data, columns=['Column Name', 'a', 'b', 'd', 'e', 'f',
                                          'h', 'i', 'j', 'k', 'l'])
        assert_frame_equal(df1, df2)

        np.random.seed(1)
        a = np.random.randint(0, 100, (10, 5))
        df = de.DataFrame(a)
        df1 = df.corr()
        data = {'Column Name': ['a0', 'a1', 'a2', 'a3', 'a4'],
                'a0': [1.0,
                       0.1878981809143684,
                       -0.04913753994096178,
                       0.30624690075804206,
                       0.08573019636314326],
                'a1': [0.1878981809143684,
                       1.0,
                       -0.09201870004461146,
                       0.14317713312049318,
                       -0.45649118427652063],
                'a2': [-0.04913753994096178,
                       -0.09201870004461146,
                       1.0,
                       -0.14938446325950921,
                       -0.4040167085005531],
                'a3': [0.30624690075804206,
                       0.14317713312049318,
                       -0.14938446325950921,
                       1.0,
                       -0.005768700947290365],
                'a4': [0.08573019636314326,
                       -0.45649118427652063,
                       -0.4040167085005531,
                       -0.005768700947290365,
                       1.0]}
        df2 = de.DataFrame(data, columns=['Column Name', 'a0', 'a1', 'a2', 'a3', 'a4'])
        assert_frame_equal(df1, df2)


class TestSummary(object):

    def test_quantile(self):
        data = {'a': [0, 0, 5, 9, 3, 4, 5, 1],
                'b': [0, 1.512344353, 8, 9, np.nan, 3, 2, 8],
                'c': [''] + list('bgggzgh'),
                'd': [False, False, True, False] * 2,
                'e': [0, 20, 30, 4, 5, 6, 7, 8],
                'f': [0., 3, 3, 3, 11, 4, 5, 1],
                'g': ['', None, 'ad', 'effd', 'ef', None, 'ett', 'zzzz'],
                'h': [0, 4, 5, 6, 7, 8, 9, 0],
                'i': np.array([0, 7, 6, 5, 4, 3, 2, 11]),
                'j': np.zeros(8, dtype='int64'),
                'k': np.ones(8) - 1,
                'l': [np.nan] * 8}

        df = de.DataFrame(data)
        df1 = df.quantile(q=.5)
        df2 = de.DataFrame({'a': [3.5],
                            'b': [3.0],
                            'd': [0.0],
                            'e': [6.5],
                            'f': [3.0],
                            'h': [5.5],
                            'i': [4.5],
                            'j': [0.0],
                            'k': [0.0],
                            'l': [nan]})
        assert_frame_equal(df1, df2)

        df1 = df.quantile(q=.8)
        df2 = de.DataFrame({'a': [5.0],
                            'b': [8.0],
                            'd': [0.6],
                            'e': [15.2],
                            'f': [4.6],
                            'h': [7.6],
                            'i': [6.6],
                            'j': [0.0],
                            'k': [0.0],
                            'l': [nan]})
        assert_frame_equal(df1, df2)

        df1 = df.select_dtypes('int').quantile(q=.2)
        df2 = de.DataFrame({'a': [0.4],
                            'e': [4.4],
                            'h': [1.6],
                            'i': [2.4],
                            'j': [0.0]})
        assert_frame_equal(df1, df2)

        df1 = df.quantile('columns', q=.5)
        df2 = de.DataFrame({'quantile': [0.0, 1.512344353, 5.0, 4.0, 3.5, 3.0, 2.0, 1.0]})
        assert_frame_equal(df1, df2)

        with pytest.raises(TypeError):
            df.quantile(q='asd')

        with pytest.raises(ValueError):
            df.quantile(q=1.1)

    def test_describe(self):
        data = {'a': [0, 0, 5, 9, 3, 4, 5, 1],
                'b': [0, 1.512344353, 8, 9, np.nan, 3, 2, 8],
                'c': [''] + list('bgggzgh'),
                'd': [False, False, True, False] * 2,
                'e': [0, 20, 30, 4, 5, 6, 7, 8],
                'f': [0., 3, 3, 3, 11, 4, 5, 1],
                'g': ['', nan, 'ad', 'effd', 'ef', None, 'ett', 'zzzz'],
                'h': [0, 4, 5, 6, 7, 8, 9, 0],
                'i': np.array([0, 7, 6, 5, 4, 3, 2, 11]),
                'j': np.zeros(8, dtype='int64'),
                'k': np.ones(8) - 1,
                'l': [np.nan] * 8}

        df = de.DataFrame(data)
        df1 = df.describe()
        data = {'25%': [0.75, 1.7561721765, 4.75, 2.5, 3.0, 2.75, 0.0, 0.0, nan],
                '50%': [3.5, 3.0, 6.5, 3.0, 5.5, 4.5, 0.0, 0.0, nan],
                '75%': [5.0, 8.0, 11.0, 4.25, 7.25, 6.25, 0.0, 0.0, nan],
                'Column Name': ['a', 'b', 'e', 'f', 'h', 'i', 'j', 'k', 'l'],
                'Data Type': ['int',
                              'float',
                              'int',
                              'float',
                              'int',
                              'int',
                              'int',
                              'float',
                              'float'],
                'count': [8, 7, 8, 8, 8, 8, 8, 8, 0],
                'max': [9.0, 9.0, 30.0, 11.0, 9.0, 11.0, 0.0, 0.0, nan],
                'mean': [3.375, 4.501763479, 10.0, 3.75, 4.875, 4.75, 0.0, 0.0, nan],
                'min': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, nan],
                'null %': [0.0, 0.125, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                'std': [3.0676887530703447,
                        3.7064372942230235,
                        9.92831448793946,
                        3.3273756282434617,
                        3.399054490379851,
                        3.370036032024414,
                        0.0,
                        0.0,
                        nan]}
        df2 = de.DataFrame(data, columns=['Column Name',
                                          'Data Type',
                                          'count',
                                          'null %',
                                          'mean',
                                          'std',
                                          'min',
                                          '25%',
                                          '50%',
                                          '75%',
                                          'max'])
        assert_frame_equal(df1, df2)

        df1 = df.describe(percentiles=[.2, .1, .4, .99])
        data = {'10%': [0.0,
                        0.9074066118000002,
                        2.8000000000000003,
                        0.7000000000000001,
                        0.0,
                        1.4000000000000001,
                        0.0,
                        0.0,
                        nan],
                '20%': [0.40000000000000013,
                        1.6098754824000001,
                        4.4,
                        1.8000000000000003,
                        1.6000000000000005,
                        2.4000000000000004,
                        0.0,
                        0.0,
                        nan],
                '40%': [2.6000000000000005,
                        2.4000000000000004,
                        5.800000000000001,
                        3.0,
                        4.800000000000001,
                        3.8000000000000003,
                        0.0,
                        0.0,
                        nan],
                '99%': [8.719999999999999,
                        8.94,
                        29.299999999999997,
                        10.579999999999998,
                        8.93,
                        10.719999999999999,
                        0.0,
                        0.0,
                        nan],
                'Column Name': ['a', 'b', 'e', 'f', 'h', 'i', 'j', 'k', 'l'],
                'Data Type': ['int',
                              'float',
                              'int',
                              'float',
                              'int',
                              'int',
                              'int',
                              'float',
                              'float'],
                'count': [8, 7, 8, 8, 8, 8, 8, 8, 0],
                'max': [9.0, 9.0, 30.0, 11.0, 9.0, 11.0, 0.0, 0.0, nan],
                'mean': [3.375, 4.501763479, 10.0, 3.75, 4.875, 4.75, 0.0, 0.0, nan],
                'min': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, nan],
                'null %': [0.0, 0.125, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                'std': [3.0676887530703447,
                        3.7064372942230235,
                        9.92831448793946,
                        3.3273756282434617,
                        3.399054490379851,
                        3.370036032024414,
                        0.0,
                        0.0,
                        nan]}
        df2 = de.DataFrame(data, columns=['Column Name',
                                          'Data Type',
                                          'count',
                                          'null %',
                                          'mean',
                                          'std',
                                          'min',
                                          '20%',
                                          '10%',
                                          '40%',
                                          '99%',
                                          'max'])
        assert_frame_equal(df1, df2)


class TestUnique(object):

    def test_unique(self):
        data = {'a': [0, 0, 5, 9, 3, 4, 5, 1],
                'b': [0, 1.512344353, 8, 9, np.nan, 3, 2, 8],
                'c': [''] + list('bgggzgh'),
                'd': [False, False, True, False] * 2,
                'e': [0, 20, 30, 4, 5, 6, 7, 8],
                'f': [0., 3, 3, 3, 11, 4, 5, 1],
                'g': ['', nan, 'ad', 'effd', 'ef', None, 'ett', 'zzzz'],
                'h': [0, 4, 5, 6, 7, 8, 9, 0],
                'i': np.array([0, 7, 6, 5, 4, 3, 2, 11]),
                'j': np.zeros(8, dtype='int64'),
                'k': np.ones(8) - 1,
                'l': [np.nan] * 8}

        df = de.DataFrame(data)
        df1 = df.unique('a', only_subset=True)
        df2 = de.DataFrame({'a': [0, 5, 9, 3, 4, 1]})
        assert_frame_equal(df1, df2)

        df1 = df.unique('b', only_subset=True)
        df2 = de.DataFrame({'b': np.array([0., 1.51234435, 8., 9., nan, 3., 2.])})
        assert_frame_equal(df1, df2)

        df1 = df.unique('c', only_subset=True)
        df2 = de.DataFrame({'c': np.array(['', 'b', 'g', 'z', 'h'], dtype=object)})
        assert_frame_equal(df1, df2)

        df1 = df.unique('d', only_subset=True)
        df2 = de.DataFrame({'d': np.array([False, True], dtype=bool)})
        assert_frame_equal(df1, df2)

        df1 = df.unique('l', only_subset=True)
        df2 = de.DataFrame({'l': np.array([nan])})
        assert_frame_equal(df1, df2)

    def test_unique_multiple_cols_1_dtype(self):
        data = {'a': [0, 0, 5, 5],
                'b': [8, 3.2, 8, 9],
                'c': list('bggg'),
                'd': [False, False, True, False],
                'e': [20, 20, 30, 30],
                'f': [4, 3.213, 4, 9],
                'g': list('cddd'),
                'h': [False, False, True, True]}
        df = de.DataFrame(data)

        df1 = df.unique(['a', 'e'], only_subset=True)
        df2 = de.DataFrame({'a': [0, 5],
                            'e': [20, 30]})
        assert_frame_equal(df1, df2)

        df1 = df.unique(['b', 'f'], only_subset=True)
        df2 = de.DataFrame({'b': [8, 3.2, 9],
                            'f': [4, 3.213, 9]})
        assert_frame_equal(df1, df2)

        df1 = df.unique(['c', 'g'], only_subset=True)
        df2 = de.DataFrame({'c': ['b', 'g'],
                            'g': ['c', 'd']})
        assert_frame_equal(df1, df2)

        df1 = df.unique(['d', 'h'], only_subset=True)
        df2 = de.DataFrame({'d': [False, True, False],
                            'h': [False, True, True]})
        assert_frame_equal(df1, df2)

    def test_unique_multiple_cols_dtypes(self):
        data = {'a': [0, 0, 5, 5],
                'b': [8, 3.2, 8, 8],
                'c': list('bggg'),
                'd': [False, False, True, False],
                'e': [20, 20, 30, 30],
                'f': [4, 3.213, 4, 9],
                'g': list('cddd'),
                'h': [False, False, True, True]}
        df = de.DataFrame(data)

        df1 = df.unique(['a', 'c'], only_subset=True)
        df2 = de.DataFrame({'a': [0, 0, 5],
                            'c': ['b', 'g', 'g']})
        assert_frame_equal(df1, df2)

        df1 = df.unique(['b', 'c'], only_subset=True)
        df2 = de.DataFrame({'b': [8, 3.2, 8],
                            'c': ['b', 'g', 'g']})
        assert_frame_equal(df1, df2)

        df1 = df.unique(['c', 'd'], only_subset=True)
        df2 = de.DataFrame({'c': ['b', 'g', 'g'],
                            'd': [False, False, True]})
        assert_frame_equal(df1, df2)

    def test_unique_all(self):
        data = {'a': [0, 0, 5, 5],
                'b': [8, 3.2, 8, 8],
                'c': list('bggg'),
                'd': [False, False, True, False],
                'e': [20, 20, 30, 30],
                'f': [4, 3.213, 4, 9],
                'g': list('cddd'),
                'h': [False, False, True, True]}
        df = de.DataFrame(data)

        df1 = df.unique('a')
        data = {'a': [0, 5],
                'b': [8, 8.],
                'c': list('bg'),
                'd': [False, True],
                'e': [20, 30],
                'f': [4, 4.],
                'g': list('cd'),
                'h': [False, True]}
        df2 = de.DataFrame(data)
        assert_frame_equal(df1, df2)

        df1 = df.unique()
        assert_frame_equal(df1, df)

        data = {'a': [0, 0, 5, 5],
                'b': [8, 3.2, 8, 8],
                'c': list('bggg'),
                'd': [False, False, True, False]}

        df = de.DataFrame(data)
        df1 = df.unique(['c', 'd'])
        df2 = de.DataFrame({'a': [0, 0, 5],
                            'b': [8, 3.2, 8],
                            'c': ['b', 'g', 'g'],
                            'd': [False, False, True]})
        assert_frame_equal(df1, df2)

    def test_unique_last(self):
        data = {'a': [0, 0, 5, 5],
                'b': [8, 3.2, 8, 8],
                'c': list('bggg'),
                'd': [False, False, True, False],
                'e': [20, 20, 30, 30],
                'f': [4, 3.213, 4, 9],
                'g': list('cddd'),
                'h': [False, False, True, True]}
        df = de.DataFrame(data)

        df1 = df.unique('a', keep='last')
        data = {'a': [0, 5],
                'b': [3.2, 8.],
                'c': list('gg'),
                'd': [False, False],
                'e': [20, 30],
                'f': [3.213, 9.],
                'g': list('dd'),
                'h': [False, True]}
        df2 = de.DataFrame(data)
        assert_frame_equal(df1, df2)

        df1 = df.unique(keep='last')
        assert_frame_equal(df1, df)

        data = {'a': [0, 0, 5, 5],
                'b': [8, 3.2, 8, 8],
                'c': list('bggg'),
                'd': [False, False, True, False]}

        df = de.DataFrame(data)
        df1 = df.unique(['c', 'd'], keep='last')
        df2 = de.DataFrame({'a': [0, 5, 5],
                            'b': [8, 8, 8.],
                            'c': ['b', 'g', 'g'],
                            'd': [False, True, False]})
        assert_frame_equal(df1, df2)

    def test_unique_none(self):
        data = {'a': [0, 0, 5, 5],
                'b': [8, 3.2, 8, 8],
                'c': list('bggg'),
                'd': [False, False, True, False]}

        df = de.DataFrame(data)
        df1 = df.unique(['a'], keep='none', only_subset=True)
        df2 = de.DataFrame({'a': np.empty(0, 'int64')})
        assert_frame_equal(df1, df2)

        df1 = df.unique('b', keep='none')
        df2 = de.DataFrame({'a': [0],
                            'b': [3.2],
                            'c': ['g'],
                            'd': [False]})
        assert_frame_equal(df1, df2)

        df1 = df.unique(['d', 'c'], keep='none', only_subset=True)
        df2 = de.DataFrame({'c': ['b', 'g'],
                            'd': [False, True]})
        assert_frame_equal(df1, df2)


class TestNUninque:

    def test_nunique(self):
        data = {'a': [0, 0, 5, 9, 3, 4, 5, 1],
                'b': [0, 1.512344353, 8, 9, np.nan, 3, 2, 8],
                'c': [''] + list('bgggzgh'),
                'd': [False, False, True, False] * 2,
                'e': [0, 20, 30, 4, 5, 6, 7, 8],
                'f': [0., 3, 3, 3, 11, 4, 5, 1],
                'g': ['', nan, 'ad', 'effd', 'ef', None, 'ett', 'zzzz'],
                'h': [0, 4, 5, 6, 7, 8, 9, 0],
                'i': np.array([0, 7, 6, 5, 4, 3, 2, 11]),
                'j': np.zeros(8, dtype='int64'),
                'k': np.ones(8) - 1,
                'l': [np.nan] * 8}
        columns = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l']

        df = de.DataFrame(data, columns)
        df1 = df.nunique()
        df2 = de.DataFrame({'a': [6],
                            'b': [6],
                            'c': [5],
                            'd': [2],
                            'e': [8],
                            'f': [6],
                            'g': [6],
                            'h': [7],
                            'i': [8],
                            'j': [1],
                            'k': [1],
                            'l': [0]}, columns)
        assert_frame_equal(df1, df2)

        df1 = df.nunique(count_na=True)
        df2 = de.DataFrame({'a': [6],
                            'b': [7],
                            'c': [5],
                            'd': [2],
                            'e': [8],
                            'f': [6],
                            'g': [7],
                            'h': [7],
                            'i': [8],
                            'j': [1],
                            'k': [1],
                            'l': [1]})
        assert_frame_equal(df1, df2)

        df1 = df.nunique('columns')
        df2 = de.DataFrame({'nunique': [2, 7, 9, 8, 8, 6, 8, 6]})
        assert_frame_equal(df1, df2)

        df1 = df.nunique('columns', count_na=True)
        df2 = de.DataFrame({'nunique': [3, 8, 10, 9, 9, 7, 9, 7]})
        assert_frame_equal(df1, df2)
