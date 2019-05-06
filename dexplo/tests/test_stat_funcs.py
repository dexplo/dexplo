import dexplo as dx
import numpy as np
from numpy import nan, array
import pytest
from dexplo.testing import assert_frame_equal


class TestSimpleAggs(object):
    df = dx.DataFrame({'a': [0, 0, 6],
                       'b': [0, 1.5, np.nan],
                       'c': [''] + list('bg'),
                       'd': [False, False, True],
                       'e': [0, 24, 30],
                       'f': ['', None, 'ad'],
                       'g': np.zeros(3, dtype='int64'),
                       'h': [np.nan] * 3,
                       'i': np.array([2, 4, 5], dtype='datetime64[ns]'),
                       'j': np.array([10, 40, -12], dtype='timedelta64[ns]'),
                       'k': np.array([200, 4, 5], dtype='datetime64[ns]'),
                       'l': np.array([20, 4, 51], dtype='timedelta64[ns]')
                       })

    df1 = dx.DataFrame({'a': [0, 0, 6, 6],
                        'b': [0, 1.5, np.nan, 2],
                        'c': ['b', 'b', 'g', 'a'],
                        'd': [False, False, True, True],
                        'e': [0, 24, 24, 0],
                        'f': ['', None, 'ad', 'wer'],
                        'g': np.zeros(4, dtype='int64'),
                        'h': [np.nan] * 4,
                        'i': np.array([2, 4, 4, 2], dtype='datetime64[ns]'),
                        'j': np.array([10, 40, 10, 40], dtype='timedelta64[ns]'),
                        'k': np.array([200, 4, 5, -99], dtype='datetime64[ns]'),
                        'l': np.array([20, 4, 51, 4], dtype='timedelta64[ns]')
                        })

    def test_sum_vertical(self):
        df1 = self.df.sum()
        df2 = dx.DataFrame({'a': [6],
                            'b': [1.5],
                            'd': [1],
                            'e': [54],
                            'g': [0],
                            'h': [0.],
                            'j': np.array([38], dtype='timedelta64[ns]'),
                            'l': np.array([75], dtype='timedelta64[ns]')})
        assert_frame_equal(df1, df2)

        df1 = self.df.sum(include_strings=True)
        df2 = dx.DataFrame({'a': [6],
                            'b': [1.5],
                            'c': ['bg'],
                            'd': [1],
                            'e': [54],
                            'f': ['ad'],
                            'g': [0],
                            'h': [0.],
                            'j': np.array([38], dtype='timedelta64[ns]'),
                            'l': np.array([75], dtype='timedelta64[ns]')})
        assert_frame_equal(df1, df2)

        with pytest.raises(ValueError):
            self.df.select_dtypes('str').sum()

    def test_sum_horizontal(self):

        with pytest.raises(TypeError):
            self.df.sum(axis='columns')

        df = self.df.select_dtypes('number')
        df1 = df.sum(axis='columns')
        df2 = dx.DataFrame({'sum': array([0., 25.5, 36.])})
        assert_frame_equal(df1, df2)

        df1 = self.df.select_dtypes('str').sum(axis='columns', include_strings=True)
        df2 = dx.DataFrame({'sum': ['', 'b', 'gad']})
        assert_frame_equal(df1, df2)

    def test_max_vertical(self):
        df1 = self.df.select_dtypes(exclude='str').max()
        df2 = dx.DataFrame({'a': [6],
                            'b': [1.5],
                            'd': [1],
                            'e': [30],
                            'g': [0],
                            'h': [np.nan],
                            'i': np.array([5], dtype='datetime64[ns]'),
                            'j': np.array([40], dtype='timedelta64[ns]'),
                            'k': np.array([200], dtype='datetime64[ns]'),
                            'l': np.array([51], dtype='timedelta64[ns]')
                            })
        assert_frame_equal(df1, df2)

        df1 = self.df.max()
        df2 = dx.DataFrame({'a': [6],
                            'b': [1.5],
                            'c': ['g'],
                            'd': [1],
                            'e': [30],
                            'f': ['ad'],
                            'g': [0],
                            'h': [np.nan],
                            'i': np.array([5], dtype='datetime64[ns]'),
                            'j': np.array([40], dtype='timedelta64[ns]'),
                            'k': np.array([200], dtype='datetime64[ns]'),
                            'l': np.array([51], dtype='timedelta64[ns]')
                            })
        assert_frame_equal(df1, df2)

    def test_max_horizontal(self):

        with pytest.raises(TypeError):
            self.df.max(axis='columns')

        df1 = self.df.select_dtypes('number').max(axis='columns')
        df2 = dx.DataFrame({'max': [0, 24., 30]})
        assert_frame_equal(df1, df2)

        df1 = self.df.select_dtypes('str').max(axis='columns')
        df2 = dx.DataFrame({'max': ['', 'b', 'g']})
        assert_frame_equal(df1, df2)

        df1 = self.df.select_dtypes('timedelta').max(axis='columns')
        df2 = dx.DataFrame({'max': np.array([20, 40, 51], dtype='timedelta64[ns]')})
        assert_frame_equal(df1, df2)

    def test_min_vertical(self):
        df1 = self.df.select_dtypes(exclude='str').min()
        df2 = dx.DataFrame({'a': [0],
                            'b': [0.],
                            'd': [0],
                            'e': [0],
                            'g': [0],
                            'h': [np.nan],
                            'i': np.array([2], dtype='datetime64[ns]'),
                            'j': np.array([-12], dtype='timedelta64[ns]'),
                            'k': np.array([4], dtype='datetime64[ns]'),
                            'l': np.array([4], dtype='timedelta64[ns]')
                            })
        assert_frame_equal(df1, df2)

        df1 = self.df.min()
        df2 = dx.DataFrame({'a': [0],
                            'b': [0.],
                            'c': [''],
                            'd': [0],
                            'e': [0],
                            'f': [''],
                            'g': [0],
                            'h': [np.nan],
                            'i': np.array([2], dtype='datetime64[ns]'),
                            'j': np.array([-12], dtype='timedelta64[ns]'),
                            'k': np.array([4], dtype='datetime64[ns]'),
                            'l': np.array([4], dtype='timedelta64[ns]')
                            })
        assert_frame_equal(df1, df2)

    def test_min_horizontal(self):

        with pytest.raises(TypeError):
            self.df.min(axis='columns')

        df1 = self.df.select_dtypes('number').min(axis='columns')
        df2 = dx.DataFrame({'min': [0., 0, 0]})
        assert_frame_equal(df1, df2)

    def test_mean_vertical(self):
        df1 = self.df.mean()
        df2 = dx.DataFrame({'a': [2.],
                            'b': [.75],
                            'd': [1 / 3],
                            'e': [18.],
                            'g': [0.],
                            'h': [np.nan],
                            'j': np.array([12], dtype='timedelta64[ns]'),
                            'l': np.array([25], dtype='timedelta64[ns]')
                            })
        assert_frame_equal(df1, df2)

        with pytest.raises(ValueError):
            self.df.select_dtypes('str').mean()

    def test_mean_horizontal(self):
        with pytest.raises(TypeError):
            self.df.mean(axis='columns')

        df1 = self.df.select_dtypes('number').mean(axis='columns')
        df2 = dx.DataFrame({'mean': [0, 6.375, 12]})
        assert_frame_equal(df1, df2)

        df1 = self.df.select_dtypes(['timedelta']).mean(axis='columns')
        df2 = dx.DataFrame({'mean': np.array([15, 22, 19], dtype='timedelta64[ns]')})
        assert_frame_equal(df1, df2)

    def test_median_vertical(self):
        df1 = self.df.median()
        df2 = dx.DataFrame({'a': array([0.]),
                            'b': array([0.75]),
                            'd': array([0.]),
                            'e': array([24.]),
                            'g': array([0.]),
                            'h': array([nan]),
                            'j': array([10], dtype='timedelta64[ns]'),
                            'l': array([20], dtype='timedelta64[ns]')})
        assert_frame_equal(df1, df2)

        with pytest.raises(ValueError):
            self.df.select_dtypes('str').median()

    def test_median_horizontal(self):
        with pytest.raises(TypeError):
            self.df.median(axis='columns')

        df1 = self.df.select_dtypes('number').median(axis='columns')
        df2 = dx.DataFrame({'median': array([0., 0.75, 6.])})
        assert_frame_equal(df1, df2)

    def test_mode_low(self):
        df1 = self.df1.select_dtypes(exclude='str').mode(keep='low')
        df2 = dx.DataFrame({'a': array([0]),
                            'b': array([0.]),
                            'd': array([False]),
                            'e': array([0]),
                            'g': array([0]),
                            'h': array([nan]),
                            'i': array(['1970-01-01T00:00:00.000000002'], dtype='datetime64[ns]'),
                            'j': array([10], dtype='timedelta64[ns]'),
                            'k': array(['1969-12-31T23:59:59.999999901'], dtype='datetime64[ns]'),
                            'l': array([4], dtype='timedelta64[ns]')})
        assert_frame_equal(df1, df2)
        df1 = self.df1.mode(keep='low')
        df2 = dx.DataFrame({'a': array([0]),
                            'b': array([0.]),
                            'c': array(['b']),
                            'd': array([False]),
                            'e': array([0]),
                            'f': array(['']),
                            'g': array([0]),
                            'h': array([nan]),
                            'i': array(['1970-01-01T00:00:00.000000002'], dtype='datetime64[ns]'),
                            'j': array([10], dtype='timedelta64[ns]'),
                            'k': array(['1969-12-31T23:59:59.999999901'], dtype='datetime64[ns]'),
                            'l': array([4], dtype='timedelta64[ns]')})
        assert_frame_equal(df1, df2)

        df1 = self.df1.select_dtypes('number').mode(axis='columns')
        df2 = dx.DataFrame({'mode': array([0., 0., 0., 0.])})
        assert_frame_equal(df1, df2)

    def test_mode_high(self):
        df1 = self.df1.select_dtypes(exclude='str').mode(keep='high')
        df2 = dx.DataFrame({'a': array([6]),
                            'b': array([2.]),
                            'd': array([True]),
                            'e': array([24]),
                            'g': array([0]),
                            'h': array([nan]),
                            'i': array(['1970-01-01T00:00:00.000000004'], dtype='datetime64[ns]'),
                            'j': array([40], dtype='timedelta64[ns]'),
                            'k': array(['1970-01-01T00:00:00.000000200'], dtype='datetime64[ns]'),
                            'l': array([4], dtype='timedelta64[ns]')})
        assert_frame_equal(df1, df2)

        df1 = self.df1.select_dtypes('number').mode(keep='high', axis='columns')
        df2 = dx.DataFrame({'mode': array([ 0.,  0., 24.,  0.])})
        assert_frame_equal(df1, df2)

    def test_std(self):
        df1 = self.df1.std()
        df2 = dx.DataFrame({'a': array([3.46410162]),
                            'b': array([1.040833]),
                            'd': array([0.57735027]),
                            'e': array([13.85640646]),
                            'g': array([0.]),
                            'h': array([nan])})
        assert_frame_equal(df1, df2)

        df1 = self.df1.select_dtypes('number').std(axis='columns')
        df2 = dx.DataFrame({'std': array([ 0., 11.77125737, 12.489996, 2.82842712])})
        assert_frame_equal(df1, df2)

        with pytest.raises(TypeError):
            self.df1.std(axis='columns')

    def test_var(self):

        df1 = self.df1.var()
        df2 = dx.DataFrame({'a': array([12.]),
                             'b': array([1.08333333]),
                             'd': array([0.33333333]),
                             'e': array([192.]),
                             'g': array([0.]),
                             'h': array([nan])})
        assert_frame_equal(df1, df2)

        df1 = self.df1.select_dtypes('number').var(axis='columns')
        df2 = dx.DataFrame({'var': array([ 0., 138.5625, 156., 8. ])})
        assert_frame_equal(df1, df2)

    def test_prod(self):
        df1 = self.df1.prod()
        df2 = dx.DataFrame({'a': array([0]),
                            'b': array([0.]),
                            'd': array([0]),
                            'e': array([0]),
                            'g': array([0]),
                            'h': array([1.])})
        assert_frame_equal(df1, df2)

        df1 = self.df1.select_dtypes('number').prod(axis='columns')
        df2 = dx.DataFrame({'prod': array([0., 0., 0., 0.])})
        assert_frame_equal(df1, df2)

    def test_count(self):
        df1 = self.df1.count()
        df2 = dx.DataFrame({'a': array([4]),
                            'b': array([3]),
                            'c': array([4]),
                            'd': array([4]),
                            'e': array([4]),
                            'f': array([3]),
                            'g': array([4]),
                            'h': array([0]),
                            'i': array([4]),
                            'j': array([4]),
                            'k': array([4]),
                            'l': array([4])})
        assert_frame_equal(df1, df2)

        df1 = self.df1.count(axis='columns')
        df2 = dx.DataFrame({'count': array([11, 10, 10, 11])})
        assert_frame_equal(df1, df2)


class TestAnyAll(object):

    df1 = dx.DataFrame({'a': [0, 0, 6, 6],
                        'b': [0, 1.5, np.nan, 2],
                        'c': ['b', 'b', 'g', 'a'],
                        'd': [False, False, True, True],
                        'e': [0, 24, 24, 0],
                        'f': ['', None, 'ad', 'wer'],
                        'g': np.zeros(4, dtype='int64'),
                        'h': [np.nan] * 4,
                        'i': np.array([2, 4, 4, 2], dtype='datetime64[ns]'),
                        'j': np.array([10, 40, 10, 40], dtype='timedelta64[ns]'),
                        'k': np.array([200, 4, 5, -99], dtype='datetime64[ns]'),
                        'l': np.array([20, 4, 51, 4], dtype='timedelta64[ns]')
                        })

    def test_any(self):
        df1 = self.df1.any()
        df2 = dx.DataFrame({'a': array([ True]),
                            'b': array([ True]),
                            'c': array([ True]),
                            'd': array([ True]),
                            'e': array([ True]),
                            'f': array([ True]),
                            'g': array([False]),
                            'h': array([False]),
                            'i': array([ True]),
                            'j': array([ True]),
                            'k': array([ True]),
                            'l': array([ True])})
        assert_frame_equal(df1, df2)

        with pytest.raises(TypeError):
            self.df1.any(axis='columns')

    def test_all(self):
        df1 = self.df1.all()
        df2 = dx.DataFrame({'a': array([False]),
                            'b': array([False]),
                            'c': array([ True]),
                            'd': array([False]),
                            'e': array([False]),
                            'f': array([False]),
                            'g': array([False]),
                            'h': array([False]),
                            'i': array([ True]),
                            'j': array([ True]),
                            'k': array([ True]),
                            'l': array([ True])})
        assert_frame_equal(df1, df2)

        with pytest.raises(TypeError):
            self.df1.all(axis='columns')


class TestArgMinMax(object):

    df1 = dx.DataFrame({'a': [0, 0, 6, 6],
                        'b': [0, 1.5, np.nan, 2],
                        'c': ['b', 'b', 'g', 'a'],
                        'd': [False, False, True, True],
                        'e': [0, 24, 24, 0],
                        'f': ['', None, 'ad', 'wer'],
                        'g': np.zeros(4, dtype='int64'),
                        'h': [np.nan] * 4,
                        'i': np.array([2, 4, 4, 2], dtype='datetime64[ns]'),
                        'j': np.array([10, 40, 10, 40], dtype='timedelta64[ns]'),
                        'k': np.array([200, 4, 5, -99], dtype='datetime64[ns]'),
                        'l': np.array([20, 4, 51, 4], dtype='timedelta64[ns]')
                        })

    def test_argmax(self):
        df1 = self.df1.argmax()
        df2 = dx.DataFrame({'a': array([2]),
                            'b': array([3.]),
                            'c': array([2]),
                            'd': array([2]),
                            'e': array([1]),
                            'f': array([3]),
                            'g': array([0]),
                            'h': array([nan]),
                            'i': array([1]),
                            'j': array([1]),
                            'k': array([0]),
                            'l': array([2])})
        assert_frame_equal(df1, df2)

        with pytest.raises(TypeError):
            self.df1.argmax(axis='columns')

        df1 = self.df1.select_dtypes('number').argmax(axis='columns')
        df2 = dx.DataFrame({'argmax': array([0, 2, 2, 0])})
        assert_frame_equal(df1, df2)

    def test_argmin(self):
        df1 = self.df1.argmin()
        df2 = dx.DataFrame({'a': array([0]),
                            'b': array([0.]),
                            'c': array([3]),
                            'd': array([0]),
                            'e': array([0]),
                            'f': array([0]),
                            'g': array([0]),
                            'h': array([nan]),
                            'i': array([0]),
                            'j': array([0]),
                            'k': array([3]),
                            'l': array([1])})
        assert_frame_equal(df1, df2)

        with pytest.raises(TypeError):
            self.df1.argmin(axis='columns')

        df1 = self.df1.select_dtypes('number').argmin(axis='columns')
        df2 = dx.DataFrame({'argmin': array([0, 0, 3, 2])})
        assert_frame_equal(df1, df2)


class TestAbsRoundClip(object):
    df = dx.DataFrame({'a': [0, -109, 1234, 603],
                       'b': [0.19185, -1.5123, np.nan, 122.445],
                       'c': ['b', 'b', 'g', 'a'],
                       'd': [False, False, True, True],
                       'e': [-9981, 2411, 2423, -123],
                       'f': ['', None, 'ad', 'wer'],
                       'g': np.zeros(4, dtype='int64'),
                       'h': [np.nan] * 4,
                       'i': np.array([2, 4, 4, 2], dtype='datetime64[ns]'),
                       'j': np.array([-10, -40, 10, 40], dtype='timedelta64[ns]'),
                       'k': np.array([200, 4, 5, -99], dtype='datetime64[ns]'),
                       'l': np.array([20, 4, -51, 4], dtype='timedelta64[ns]')
                       })

    def test_abs(self):
        df1 = self.df.abs()
        df2 = dx.DataFrame({'a': array([0,  109, 1234,  603]),
                            'b': array([0.19185,   1.5123, nan, 122.445]),
                            'c': array(['b', 'b', 'g', 'a'], dtype=object),
                            'd': array([False, False,  True,  True]),
                            'e': array([9981, 2411, 2423,  123]),
                            'f': array(['', None, 'ad', 'wer'], dtype=object),
                            'g': array([0, 0, 0, 0]),
                            'h': array([nan, nan, nan, nan]),
                            'i': array([2, 4, 4, 2], dtype='datetime64[ns]'),
                            'j': array([10, 40, 10, 40], dtype='timedelta64[ns]'),
                            'k': array([200, 4, 5, -99], dtype='datetime64[ns]'),
                            'l': array([20,  4, 51,  4], dtype='timedelta64[ns]')})
        assert_frame_equal(df1, df2)

        df1 = abs(self.df)
        assert_frame_equal(df1, df2)

    def test_round(self):
        df1 = self.df.round(1)
        df2 = dx.DataFrame({'a': [0, -109, 1234, 603],
                            'b': [0.2, -1.5, np.nan, 122.4],
                            'c': ['b', 'b', 'g', 'a'],
                            'd': [False, False, True, True],
                            'e': [-9981, 2411, 2423, -123],
                            'f': ['', None, 'ad', 'wer'],
                            'g': np.zeros(4, dtype='int64'),
                            'h': [np.nan] * 4,
                            'i': np.array([2, 4, 4, 2], dtype='datetime64[ns]'),
                            'j': np.array([-10, -40, 10, 40], dtype='timedelta64[ns]'),
                            'k': np.array([200, 4, 5, -99], dtype='datetime64[ns]'),
                            'l': np.array([20, 4, -51, 4], dtype='timedelta64[ns]')
                            })
        assert_frame_equal(df1, df2)

    def test_clip_numbers(self):
        df = dx.DataFrame({'a': [0, 0, 5],
                           'b': [0, 1.5, np.nan],
                           'c': [''] + list('bb'),
                           'd': [False, False, True],
                           'e': [90, 20, 30],
                           'f': ['', None, 'ad'],
                           'g': np.zeros(3, dtype='int64'),
                           'h': [np.nan] * 3})

        df1 = df.clip(2)
        df2 = dx.DataFrame({'a': [2, 2, 5],
                            'b': [2, 2, np.nan],
                            'c': [''] + list('bb'),
                            'd': [False, False, True],
                            'e': [90, 20, 30],
                            'f': ['', None, 'ad'],
                            'g': [2, 2, 2],
                            'h': [np.nan] * 3})
        assert_frame_equal(df1, df2)

        df1 = df.clip(upper=1)
        df2 = dx.DataFrame({'a': [0, 0, 1],
                            'b': [0, 1., np.nan],
                            'c': [''] + list('bb'),
                            'd': [False, False, True],
                            'e': [1, 1, 1],
                            'f': ['', None, 'ad'],
                            'g': np.zeros(3, dtype='int64'),
                            'h': [np.nan] * 3})
        assert_frame_equal(df1, df2)

        with pytest.raises(ValueError):
            df.clip(10, 2)

    def test_clip_strings(self):
        df = dx.DataFrame({'a': [0, 0, 5],
                           'b': [0, 1.5, np.nan],
                           'c': ['ant', 'finger', 'monster'],
                           'd': [False, False, True],
                           'e': [90, 20, 30],
                           'f': ['Deer', None, 'Table'],
                           'g': np.zeros(3, dtype='int64'),
                           'h': [np.nan] * 3})

        df1 = df.clip('bat')
        df2 = dx.DataFrame({'a': [0, 0, 5],
                            'b': [0, 1.5, np.nan],
                            'c': ['bat', 'finger', 'monster'],
                            'd': [False, False, True],
                            'e': [90, 20, 30],
                            'f': ['bat', None, 'bat'],
                            'g': np.zeros(3, dtype='int64'),
                            'h': [np.nan] * 3})
        assert_frame_equal(df1, df2)

        df1 = df.clip('bat', 'kind')
        df2 = dx.DataFrame({'a': [0, 0, 5],
                            'b': [0, 1.5, np.nan],
                            'c': ['bat', 'finger', 'kind'],
                            'd': [False, False, True],
                            'e': [90, 20, 30],
                            'f': ['bat', None, 'bat'],
                            'g': np.zeros(3, dtype='int64'),
                            'h': [np.nan] * 3})
        assert_frame_equal(df1, df2)

        with pytest.raises(ValueError):
            df.clip('Z', 'ER')

class TestAccumulate(object):

    def test_cummax(self):
        df = dx.DataFrame({'a': [0, 0, 5],
                           'b': [0, 1.5, nan],
                           'c': ['', 'g', 'b'],
                           'd': [False, False, True],
                           'e': [90, 20, 30],
                           'f': [nan, 10, 4],
                           'g': ['', None, 'ad'],
                           'h': [nan] * 3})
        df1 = df.cummax()
        df2 = dx.DataFrame({'a': [0, 0, 5],
                            'b': [0, 1.5, 1.5],
                            'c': ['', 'g', 'g'],
                            'd': [False, False, True],
                            'e': [90, 90, 90],
                            'f': [nan, 10, 10],
                            'g': ['', '', 'ad'],
                            'h': [np.nan] * 3})
        assert_frame_equal(df1, df2)

        with pytest.raises(TypeError):
            df.cummax(axis='columns')

        df1 = df.select_dtypes('number').cummax(axis='columns')
        df2 = dx.DataFrame({'a': array([0., 0., 5.]),
                            'b': array([0., 1.5, 5.]),
                            'e': array([90., 20., 30.]),
                            'f': array([90., 20., 30.]),
                            'h': array([90., 20., 30.])})

        assert_frame_equal(df1, df2)

    def test_cummin(self):
        df = dx.DataFrame({'a': [0, 0, 5],
                           'b': [0, 1.5, nan],
                           'c': ['', 'g', 'b'],
                           'd': [False, False, True],
                           'e': [90, 20, 30],
                           'f': [nan, 10, 4],
                           'g': ['', None, 'ad'],
                           'h': [nan] * 3})
        df1 = df.cummin()
        df2 = dx.DataFrame({'a': array([0, 0, 0]),
                            'b': array([0., 0., 0.]),
                            'c': array(['', '', ''], dtype=object),
                            'd': array([False, False, False]),
                            'e': array([90, 20, 20]),
                            'f': array([nan, 10.,  4.]),
                            'g': array(['', '', ''], dtype=object),
                            'h': array([nan, nan, nan])})
        assert_frame_equal(df1, df2)

        with pytest.raises(TypeError):
            df.cummin(axis='columns')

        df1 = df.select_dtypes('number').cummin(axis='columns')
        df2 = dx.DataFrame({'a': array([0., 0., 5.]),
                            'b': array([0., 0., 5.]),
                            'e': array([0., 0., 5.]),
                            'f': array([0., 0., 4.]),
                            'h': array([0., 0., 4.])})
        assert_frame_equal(df1, df2)

    def test_cumsum(self):
        df = dx.DataFrame({'a': [0, 0, 5],
                           'b': [0, 1.5, nan],
                           'c': ['', 'g', 'b'],
                           'd': [False, False, True],
                           'e': [90, 20, 30],
                           'f': [nan, 10, 4],
                           'g': ['', None, 'ad'],
                           'h': [nan] * 3})
        df1 = df.cumsum()
        df2 = dx.DataFrame({'a': [0, 0, 5],
                            'b': [0, 1.5, 1.5],
                            'c': ['', 'g', 'gb'],
                            'd': [0, 0, 1],
                            'e': [90, 110, 140],
                            'f': [0., 10, 14],
                            'g': ['', '', 'ad'],
                            'h': [0., 0, 0]})
        assert_frame_equal(df1, df2)

        df1 = df.select_dtypes('number').cumsum('columns')
        data = {'a': array([0., 0., 5.]),
                'b': array([0., 1.5, 5.]),
                'e': array([90., 21.5, 35]),
                'f': array([90., 31.5, 39.]),
                'h': array([90., 31.5, 39.])}

        df2 = dx.DataFrame(data, columns=list('abefh'))
        assert_frame_equal(df1, df2)

        df1 = df.select_dtypes('str').cumsum('columns')
        df2 = dx.DataFrame({'c': ['', 'g', 'b'],
                            'g': ['', 'g', 'bad']})
        assert_frame_equal(df1, df2)

    def test_cumprod(self):
        df = dx.DataFrame({'a': [0, 0, 5],
                           'b': [0, 1.5, nan],
                           'c': ['', 'g', 'b'],
                           'd': [False, False, True],
                           'e': [90, 20, 30],
                           'f': [nan, 10, 4],
                           'g': ['', None, 'ad'],
                           'h': [nan] * 3})
        df1 = df.cumprod()
        df2 = dx.DataFrame({'a': array([0, 0, 0]),
                            'b': array([0., 0., 0.]),
                            'c': ['', 'g', 'b'],
                            'd': [0, 0, 0],
                            'e': array([   90,  1800, 54000]),
                            'f': array([ 1., 10., 40.]),
                            'g': ['', None, 'ad'],
                            'h': array([1., 1., 1.])})
        assert_frame_equal(df1, df2)

        df1 = df.select_dtypes('number').cumprod(axis='columns')
        df2 = dx.DataFrame({'a': array([0., 0., 5.]),
                            'b': array([0., 0., 5.]),
                            'e': array([  0.,   0., 150.]),
                            'f': array([  0.,   0., 600.]),
                            'h': array([  0.,   0., 600.])})
        assert_frame_equal(df1, df2)


class TestCovCorr(object):

    def test_cov(self):
        df = dx.DataFrame({'a': [0, 0, 5, 1],
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
        df2 = dx.DataFrame(data)
        assert_frame_equal(df1, df2)

        df1 = df.select_dtypes('int').cov()
        data = {'Column Name': ['a', 'e'],
                'a': [5.666666666666667, -20.166666666666668],
                'e': [-20.166666666666668, 1476.9166666666667]}
        df2 = dx.DataFrame(data)
        assert_frame_equal(df1, df2)

        with pytest.raises(TypeError):
            df.select_dtypes('str').cov()

    def test_corr(self):
        df = dx.DataFrame({'a': [0, 0, 5, 1],
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
        df2 = dx.DataFrame(data)
        assert_frame_equal(df1, df2)

        df1 = df.select_dtypes('int').corr()
        data = {'Column Name': ['a', 'e'],
                'a': [1.0, -0.2204409585872243],
                'e': [-0.2204409585872243, 1.]}
        df2 = dx.DataFrame(data)
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

        df = dx.DataFrame(data)
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
        df2 = dx.DataFrame(data, columns=['Column Name', 'a', 'b', 'd', 'e', 'f',
                                          'h', 'i', 'j', 'k', 'l'])
        assert_frame_equal(df1, df2)

        np.random.seed(1)
        a = np.random.randint(0, 100, (10, 5))
        df = dx.DataFrame(a)
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
        df2 = dx.DataFrame(data)
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

        df = dx.DataFrame(data)
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
        df2 = dx.DataFrame(data, columns=['Column Name', 'a', 'b', 'd', 'e', 'f',
                                          'h', 'i', 'j', 'k', 'l'])
        assert_frame_equal(df1, df2)

        np.random.seed(1)
        a = np.random.randint(0, 100, (10, 5))
        df = dx.DataFrame(a)
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
        df2 = dx.DataFrame(data, columns=['Column Name', 'a0', 'a1', 'a2', 'a3', 'a4'])
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

        df = dx.DataFrame(data)
        df1 = df.quantile(q=.5)
        df2 = dx.DataFrame({'a': [3.5],
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
        df2 = dx.DataFrame({'a': [5.0],
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
        df2 = dx.DataFrame({'a': [0.4],
                            'e': [4.4],
                            'h': [1.6],
                            'i': [2.4],
                            'j': [0.0]})
        assert_frame_equal(df1, df2)

        df1 = df.select_dtypes('number').quantile('columns', q=.5)
        df2 = dx.DataFrame({'quantile': array([0, 2.25617218, 5, 4.5, 4, 3.5, 3.5, 1.])})
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

        df = dx.DataFrame(data)
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
        df2 = dx.DataFrame(data, columns=['Column Name',
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
        df2 = dx.DataFrame(data, columns=['Column Name',
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

        df = dx.DataFrame(data)
        df1 = df.unique('a', only_subset=True)
        df2 = dx.DataFrame({'a': [0, 5, 9, 3, 4, 1]})
        assert_frame_equal(df1, df2)

        df1 = df.unique('b', only_subset=True)
        df2 = dx.DataFrame({'b': np.array([0., 1.51234435, 8., 9., nan, 3., 2.])})
        assert_frame_equal(df1, df2)

        df1 = df.unique('c', only_subset=True)
        df2 = dx.DataFrame({'c': np.array(['', 'b', 'g', 'z', 'h'], dtype=object)})
        assert_frame_equal(df1, df2)

        df1 = df.unique('d', only_subset=True)
        df2 = dx.DataFrame({'d': np.array([False, True], dtype=bool)})
        assert_frame_equal(df1, df2)

        df1 = df.unique('l', only_subset=True)
        df2 = dx.DataFrame({'l': np.array([nan])})
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
        df = dx.DataFrame(data)

        df1 = df.unique(['a', 'e'], only_subset=True)
        df2 = dx.DataFrame({'a': [0, 5],
                            'e': [20, 30]})
        assert_frame_equal(df1, df2)

        df1 = df.unique(['b', 'f'], only_subset=True)
        df2 = dx.DataFrame({'b': [8, 3.2, 9],
                            'f': [4, 3.213, 9]})
        assert_frame_equal(df1, df2)

        df1 = df.unique(['c', 'g'], only_subset=True)
        df2 = dx.DataFrame({'c': ['b', 'g'],
                            'g': ['c', 'd']})
        assert_frame_equal(df1, df2)

        df1 = df.unique(['d', 'h'], only_subset=True)
        df2 = dx.DataFrame({'d': [False, True, False],
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
        df = dx.DataFrame(data)

        df1 = df.unique(['a', 'c'], only_subset=True)
        df2 = dx.DataFrame({'a': [0, 0, 5],
                            'c': ['b', 'g', 'g']})
        assert_frame_equal(df1, df2)

        df1 = df.unique(['b', 'c'], only_subset=True)
        df2 = dx.DataFrame({'b': [8, 3.2, 8],
                            'c': ['b', 'g', 'g']})
        assert_frame_equal(df1, df2)

        df1 = df.unique(['c', 'd'], only_subset=True)
        df2 = dx.DataFrame({'c': ['b', 'g', 'g'],
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
        df = dx.DataFrame(data)

        df1 = df.unique('a')
        data = {'a': [0, 5],
                'b': [8, 8.],
                'c': list('bg'),
                'd': [False, True],
                'e': [20, 30],
                'f': [4, 4.],
                'g': list('cd'),
                'h': [False, True]}
        df2 = dx.DataFrame(data)
        assert_frame_equal(df1, df2)

        df1 = df.unique()
        assert_frame_equal(df1, df)

        data = {'a': [0, 0, 5, 5],
                'b': [8, 3.2, 8, 8],
                'c': list('bggg'),
                'd': [False, False, True, False]}

        df = dx.DataFrame(data)
        df1 = df.unique(['c', 'd'])
        df2 = dx.DataFrame({'a': [0, 0, 5],
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
        df = dx.DataFrame(data)

        df1 = df.unique('a', keep='last')
        data = {'a': [0, 5],
                'b': [3.2, 8.],
                'c': list('gg'),
                'd': [False, False],
                'e': [20, 30],
                'f': [3.213, 9.],
                'g': list('dd'),
                'h': [False, True]}
        df2 = dx.DataFrame(data)
        assert_frame_equal(df1, df2)

        df1 = df.unique(keep='last')
        assert_frame_equal(df1, df)

        data = {'a': [0, 0, 5, 5],
                'b': [8, 3.2, 8, 8],
                'c': list('bggg'),
                'd': [False, False, True, False]}

        df = dx.DataFrame(data)
        df1 = df.unique(['c', 'd'], keep='last')
        df2 = dx.DataFrame({'a': [0, 5, 5],
                            'b': [8, 8, 8.],
                            'c': ['b', 'g', 'g'],
                            'd': [False, True, False]})
        assert_frame_equal(df1, df2)

    def test_unique_none(self):
        data = {'a': [0, 0, 5, 5],
                'b': [8, 3.2, 8, 8],
                'c': list('bggg'),
                'd': [False, False, True, False]}

        df = dx.DataFrame(data)
        df1 = df.unique(['a'], keep='none', only_subset=True)
        df2 = dx.DataFrame({'a': np.empty(0, 'int64')})
        assert_frame_equal(df1, df2)

        df1 = df.unique('b', keep='none')
        df2 = dx.DataFrame({'a': [0],
                            'b': [3.2],
                            'c': ['g'],
                            'd': [False]})
        assert_frame_equal(df1, df2)

        df1 = df.unique(['d', 'c'], keep='none', only_subset=True)
        df2 = dx.DataFrame({'c': ['b', 'g'],
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

        df = dx.DataFrame(data, columns)
        df1 = df.nunique()
        df2 = dx.DataFrame({'a': [6],
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
        df2 = dx.DataFrame({'a': [6],
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
        df2 = dx.DataFrame({'nunique': [2, 7, 9, 8, 8, 6, 8, 6]})
        assert_frame_equal(df1, df2)

        df1 = df.nunique('columns', count_na=True)
        df2 = dx.DataFrame({'nunique': [3, 8, 10, 9, 9, 7, 9, 7]})
        assert_frame_equal(df1, df2)
