import dexplo as dx
import numpy as np
from numpy import array, nan
import pytest
from dexplo.testing import assert_frame_equal


class TestArithmeticOperations:
    df = dx.DataFrame({'a': [0, nan, 5],
                       'b': [0, 1.5, nan],
                       'c': [''] + list('bg'),
                       'd': [nan, False, True],
                       'e': ['', None, 'ad'],
                       'f': [0, 4, 5],
                       'g': np.zeros(3, dtype='int'),
                       'h': [nan, nan, nan]})

    def test_add_int(self):
        with pytest.raises(TypeError):
            self.df + 5

        df1 = self.df.select_dtypes('int') + 5
        df2 = dx.DataFrame({'a': [5, nan, 10],
                            'f': [5, 9, 10],
                            'g': [5, 5, 5]},
                           columns=['a', 'f', 'g'])
        assert_frame_equal(df1, df2)

        df1 = 5 + self.df.select_dtypes('int')
        assert_frame_equal(df1, df2)

        df1 = self.df.select_dtypes('number') + 5
        df2 = dx.DataFrame({'a': [5, nan, 10],
                            'b': [5, 6.5, nan],
                            'f': [5, 9, 10],
                            'g': [5, 5, 5],
                            'h': [nan, nan, nan]},
                           columns=list('abfgh'))
        assert_frame_equal(df1, df2)

        df1 = 5 + self.df.select_dtypes('number')
        assert_frame_equal(df1, df2)

        df1 = self.df.select_dtypes(['number', 'bool']) + 5
        df2 = dx.DataFrame({'a': [5, nan, 10],
                            'b': [5, 6.5, nan],
                            'd': [nan, 5, 6],
                            'f': [5, 9, 10],
                            'g': [5, 5, 5],
                            'h': [nan, nan, nan]},
                           columns=list('abdfgh'))
        assert_frame_equal(df1, df2)

        df1 = 5 + self.df.select_dtypes(['number', 'bool'])
        assert_frame_equal(df1, df2)

    def test_add_float(self):
        some_float = 5.0
        with pytest.raises(TypeError):
            self.df + some_float

        df1 = self.df.select_dtypes('int') + some_float
        df2 = dx.DataFrame({'a': [5., nan, 10],
                            'f': [5., 9, 10],
                            'g': [5., 5, 5]},
                           columns=['a', 'f', 'g'])
        assert_frame_equal(df1, df2)

        df1 = some_float + self.df.select_dtypes('int')
        assert_frame_equal(df1, df2)

        df1 = self.df.select_dtypes('number') + some_float
        df2 = dx.DataFrame({'a': [5., nan, 10],
                            'b': [5., 6.5, nan],
                            'f': [5., 9, 10],
                            'g': [5., 5, 5],
                            'h': [nan, nan, nan]},
                           columns=list('abfgh'))
        assert_frame_equal(df1, df2)

        df1 = some_float + self.df.select_dtypes('number')
        assert_frame_equal(df1, df2)

        df1 = self.df.select_dtypes(['number', 'bool']) + some_float
        df2 = dx.DataFrame({'a': [5., nan, 10],
                            'b': [5., 6.5, nan],
                            'd': [nan, 5., 6],
                            'f': [5., 9, 10],
                            'g': [5., 5, 5],
                            'h': [nan, nan, nan]},
                           columns=list('abdfgh'))
        assert_frame_equal(df1, df2)

        df1 = some_float + self.df.select_dtypes(['number', 'bool'])
        assert_frame_equal(df1, df2)

    def test_add_bool(self):
        some_bool = True
        with pytest.raises(TypeError):
            self.df + some_bool

        df1 = self.df.select_dtypes('int') + some_bool
        df2 = dx.DataFrame({'a': [1, nan, 6],
                            'f': [1, 5, 6],
                            'g': [1, 1, 1]},
                           columns=['a', 'f', 'g'])
        assert_frame_equal(df1, df2)

        df1 = some_bool + self.df.select_dtypes('int')
        assert_frame_equal(df1, df2)

        df1 = self.df.select_dtypes('number') + some_bool
        df2 = dx.DataFrame({'a': [1, nan, 6],
                            'b': [1, 2.5, nan],
                            'f': [1, 5, 6],
                            'g': [1, 1, 1],
                            'h': [nan, nan, nan]},
                           columns=list('abfgh'))
        assert_frame_equal(df1, df2)

        df1 = some_bool + self.df.select_dtypes('number')
        assert_frame_equal(df1, df2)

        df1 = self.df.select_dtypes(['number', 'bool']) + some_bool
        df2 = dx.DataFrame({'a': [1, nan, 6],
                            'b': [1, 2.5, nan],
                            'd': [nan, 1, 2],
                            'f': [1, 5, 6],
                            'g': [1, 1, 1],
                            'h': [nan, nan, nan]},
                           columns=list('abdfgh'))
        assert_frame_equal(df1, df2)

        df1 = some_bool + self.df.select_dtypes(['number', 'bool'])
        assert_frame_equal(df1, df2)

    def test_add_string(self):
        df1 = self.df.select_dtypes('str') + 'aaa'
        df2 = dx.DataFrame({'c': ['aaa', 'baaa', 'gaaa'],
                            'e': ['aaa', None, 'adaaa']})
        assert_frame_equal(df1, df2)

        df1 = 'aaa' + self.df.select_dtypes('str')
        df2 = dx.DataFrame({'c': ['aaa', 'aaab', 'aaag'],
                            'e': ['aaa', None, 'aaaad']})
        assert_frame_equal(df1, df2)

    def test_comparison_string(self):
        df1 = self.df.select_dtypes('str') > 'boo'
        df2 = dx.DataFrame({'c': [False, False, True],
                            'e': [False, nan, False]})
        assert_frame_equal(df1, df2)

        df1 = self.df.select_dtypes('str') < 'boo'
        df2 = dx.DataFrame({'c': [True, True, False],
                            'e': [True, nan, True]})
        assert_frame_equal(df1, df2)

        df1 = self.df.select_dtypes('str') == 'b'
        df2 = dx.DataFrame({'c': [False, True, False],
                            'e': [False, nan, False]})
        assert_frame_equal(df1, df2)

    def test_sub_int(self):
        with pytest.raises(TypeError):
            self.df - 5

        with pytest.raises(TypeError):
            self.df.select_dtypes('str') - 10

        df1 = self.df.select_dtypes('int') - 5
        df2 = dx.DataFrame({'a': [-5, nan, 0],
                            'f': [-5, -1, 0],
                            'g': [-5, -5, -5]},
                           columns=['a', 'f', 'g'])
        assert_frame_equal(df1, df2)

        df1 = 5 - self.df.select_dtypes('int')
        df2 = dx.DataFrame({'a': [5, nan, 0],
                            'f': [5, 1, 0],
                            'g': 5 - np.zeros(3, dtype='int')},
                           columns=['a', 'f', 'g'])
        assert_frame_equal(df1, df2)

        df1 = self.df.select_dtypes(['number', 'bool']) - 5
        df2 = dx.DataFrame({'a': [-5, nan, 0],
                            'b': [-5, -3.5, nan],
                            'd': [nan, -5, -4],
                            'f': [-5, -1, 0],
                            'g': np.zeros(3, dtype='int') - 5,
                            'h': [nan, nan, nan]},
                           columns=list('abdfgh'))
        assert_frame_equal(df1, df2)

        df1 = 5 - self.df.select_dtypes(['number', 'bool'])
        df2 = dx.DataFrame({'a': [5, nan, 0],
                            'b': [5, 3.5, nan],
                            'd': [nan, 5, 4],
                            'f': [5, 1, 0],
                            'g': 5 - np.zeros(3, dtype='int'),
                            'h': [nan, nan, nan]},
                           columns=list('abdfgh'))
        assert_frame_equal(df1, df2)

    def test_sub_float(self):
        some_float = 5.0
        with pytest.raises(TypeError):
            self.df - some_float

        with pytest.raises(TypeError):
            self.df.select_dtypes('str') - some_float

        df1 = self.df.select_dtypes('int') - some_float
        df2 = dx.DataFrame({'a': [-5., nan, 0],
                            'f': [-5., -1, 0],
                            'g': [-5., -5, -5]},
                           columns=['a', 'f', 'g'])
        assert_frame_equal(df1, df2)

        df1 = some_float - self.df.select_dtypes('int')
        df2 = dx.DataFrame({'a': [5., nan, 0],
                            'f': [5., 1, 0],
                            'g': 5. - np.zeros(3, dtype='int')},
                           columns=['a', 'f', 'g'])
        assert_frame_equal(df1, df2)

        df1 = self.df.select_dtypes(['number', 'bool']) - some_float
        df2 = dx.DataFrame({'a': [-5., nan, 0],
                            'b': [-5., -3.5, nan],
                            'd': [nan, -5., -4],
                            'f': [-5., -1, 0],
                            'g': np.zeros(3, dtype='int') - some_float,
                            'h': [nan, nan, nan]},
                           columns=list('abdfgh'))
        assert_frame_equal(df1, df2)

        df1 = some_float - self.df.select_dtypes(['number', 'bool'])
        df2 = dx.DataFrame({'a': [5., nan, 0],
                            'b': [5., 3.5, nan],
                            'd': [nan, 5., 4],
                            'f': [5., 1, 0],
                            'g': 5. - np.zeros(3, dtype='int'),
                            'h': [nan, nan, nan]},
                           columns=list('abdfgh'))
        assert_frame_equal(df1, df2)

    def test_sub_bool(self):
        some_bool = True
        with pytest.raises(TypeError):
            self.df + some_bool

        df1 = self.df.select_dtypes('int') - some_bool
        df2 = dx.DataFrame({'a': [-1, nan, 4],
                            'f': [-1, 3, 4],
                            'g': [-1, -1, -1]},
                           columns=['a', 'f', 'g'])
        assert_frame_equal(df1, df2)

        df1 = some_bool - self.df.select_dtypes('int')
        df2 = dx.DataFrame({'a': [1, nan, -4],
                            'f': [1, -3, -4],
                            'g': [1, 1, 1]},
                           columns=['a', 'f', 'g'])
        assert_frame_equal(df1, df2)

        df1 = self.df.select_dtypes('number') - some_bool
        df2 = dx.DataFrame({'a': [-1, nan, 4],
                            'b': [-1, .5, nan],
                            'f': [-1, 3, 4],
                            'g': [-1, -1, -1],
                            'h': [nan, nan, nan]},
                           columns=list('abfgh'))
        assert_frame_equal(df1, df2)

        df1 = some_bool - self.df.select_dtypes('number')
        df2 = dx.DataFrame({'a': [1, nan, -4],
                            'b': [1, -.5, nan],
                            'f': [1, -3, -4],
                            'g': [1, 1, 1],
                            'h': [nan, nan, nan]},
                           columns=list('abfgh'))
        assert_frame_equal(df1, df2)

        df1 = self.df.select_dtypes(['number', 'bool']) - some_bool
        df2 = dx.DataFrame({'a': [-1, nan, 4],
                            'b': [-1, .5, nan],
                            'd': [nan, -1, 0],
                            'f': [-1, 3, 4],
                            'g': [-1, -1, -1],
                            'h': [nan, nan, nan]},
                           columns=list('abdfgh'))
        assert_frame_equal(df1, df2)

        df1 = some_bool - self.df.select_dtypes(['number', 'bool'])
        df2 = dx.DataFrame({'a': [1, nan, -4],
                            'b': [1, -.5, nan],
                            'd': [nan, 1, 0],
                            'f': [1, -3, -4],
                            'g': [1, 1, 1],
                            'h': [nan, nan, nan]},
                           columns=list('abdfgh'))
        assert_frame_equal(df1, df2)

    def test_mul_int(self):
        df1 = self.df * 2
        df2 = dx.DataFrame({'a': [0, nan, 10],
                            'b': [0, 3.0, nan],
                            'c': ['', 'bb', 'gg'],
                            'd': [nan, 0, 2],
                            'e': ['', None, 'adad'],
                            'f': [0, 8, 10],
                            'g': np.zeros(3, dtype='int'),
                            'h': [nan] * 3})
        assert_frame_equal(df1, df2)

        df1 = 2 * self.df
        assert_frame_equal(df1, df2)

    def test_mul_float(self):
        some_float = 2.
        df1 = self.df * some_float
        df2 = dx.DataFrame({'a': [0., nan, 10],
                            'b': [0., 3.0, nan],
                            'c': ['', 'bb', 'gg'],
                            'd': [nan, 0., 2],
                            'e': ['', None, 'adad'],
                            'f': [0, 8., 10],
                            'g': np.zeros(3, dtype='float64'),
                            'h': [nan] * 3})
        assert_frame_equal(df1, df2)

        df1 = some_float * self.df
        assert_frame_equal(df1, df2)

    def test_mul_bool(self):
        some_bool = True
        with pytest.raises(TypeError):
            self.df * some_bool
        
        df1 = self.df.select_dtypes(['number', 'bool']) * some_bool
        df2 = dx.DataFrame({'a': [0, nan, 5],
                            'b': [0, 1.5, nan],
                            'd': [nan, 0, 1],
                            'f': [0, 4, 5],
                            'g': np.zeros(3, dtype='int'),
                            'h': [nan, nan, nan]})
        assert_frame_equal(df1, df2)

        df1 = some_bool * self.df.select_dtypes(['number', 'bool'])
        assert_frame_equal(df1, df2)

    def test_truediv_int(self):
        with pytest.raises(TypeError):
            self.df / 5

        with pytest.raises(TypeError):
            self.df.select_dtypes('str') / 10

        with pytest.raises(TypeError):
            self.df / 'asdf'

        df1 = self.df.select_dtypes('number') / 2
        df2 = dx.DataFrame({'a': [0, nan, 2.5],
                            'b': [0, .75, nan],
                            'f': [0, 2, 2.5],
                            'g': np.zeros(3),
                            'h': [nan, nan, nan]},
                           columns=list('abfgh'))
        assert_frame_equal(df1, df2)

        df1 = 10 / self.df.select_dtypes('number')
        df2 = dx.DataFrame({'a': [np.inf, nan, 2],
                            'b': [np.inf, 10 / 1.5, nan],
                            'f': [np.inf, 2.5, 2],
                            'g': [np.inf] * 3,
                            'h': [nan, nan, nan]},
                           columns=list('abfgh'))
        assert_frame_equal(df1, df2)

    def test_truediv_float(self):
        with pytest.raises(TypeError):
            self.df / 5.

        with pytest.raises(TypeError):
            self.df.select_dtypes('str') / 10.

        with pytest.raises(TypeError):
            self.df / 'asdf'

        df1 = self.df.select_dtypes('number') / 2.
        df2 = dx.DataFrame({'a': [0, nan, 2.5],
                            'b': [0, .75, nan],
                            'f': [0, 2, 2.5],
                            'g': np.zeros(3),
                            'h': [nan, nan, nan]},
                           columns=list('abfgh'))
        assert_frame_equal(df1, df2)

        df1 = 10. / self.df.select_dtypes('number')
        df2 = dx.DataFrame({'a': [np.inf, nan, 2],
                            'b': [np.inf, 10 / 1.5, nan],
                            'f': [np.inf, 2.5, 2],
                            'g': [np.inf] * 3,
                            'h': [nan, nan, nan]},
                           columns=list('abfgh'))
        assert_frame_equal(df1, df2)

    def test_truediv_bool(self):
        some_bool = True

        with pytest.raises(TypeError):
            self.df / some_bool

        with pytest.raises(TypeError):
            self.df.select_dtypes('str') / 10

        with pytest.raises(TypeError):
            self.df / 'asdf'

        df1 = self.df.select_dtypes('number')
        df2 = dx.DataFrame({'a': [0, nan, 2.5],
                            'b': [0, .75, nan],
                            'f': [0, 2, 2.5],
                            'g': np.zeros(3),
                            'h': [nan, nan, nan]},
                           columns=list('abfgh'))
        assert_frame_equal(df1, df2)

        df1 = 10 / self.df.select_dtypes('number')
        df2 = dx.DataFrame({'a': [np.inf, nan, 2],
                            'b': [np.inf, 10 / 1.5, nan],
                            'f': [np.inf, 2.5, 2],
                            'g': [np.inf] * 3,
                            'h': [nan, nan, nan]},
                           columns=list('abfgh'))
        assert_frame_equal(df1, df2)

    def test_floordiv_frame(self):
        with pytest.raises(TypeError):
            self.df // 5

        with pytest.raises(TypeError):
            self.df.select_dtypes('str') // 10

        with pytest.raises(TypeError):
            self.df // 'asdf'

        df = dx.DataFrame({'a': [0, 0, 10],
                           'b': [0, 20, nan],
                           'f': [0, 100, 10],
                           'g': np.zeros(3, dtype='int'),
                           'h': [nan, nan, nan]},
                          columns=list('abfgh'))

        df1 = df // 3
        df2 = dx.DataFrame({'a': [0, 0, 3],
                            'b': [0, 6, nan],
                            'f': [0, 33, 3],
                            'g': np.zeros(3, dtype='int'),
                            'h': [nan, nan, nan]},
                           columns=list('abfgh'))
        assert_frame_equal(df1, df2)

    def test_pow_frame(self):
        with pytest.raises(TypeError):
            self.df ** 5

        with pytest.raises(TypeError):
            self.df.select_dtypes('str') ** 10

        with pytest.raises(TypeError):
            self.df ** 'asdf'

        df = dx.DataFrame({'a': [0, 0, 10],
                           'b': [0, 2, nan],
                           'f': [0, 10, 3],
                           'g': np.zeros(3, dtype='int'),
                           'h': [nan, nan, nan]},
                          columns=list('abfgh'))
        df1 = df ** 2
        df2 = dx.DataFrame({'a': [0, 0, 100],
                            'b': [0, 4, nan],
                            'f': [0, 100, 9],
                            'g': np.zeros(3, dtype='int'),
                            'h': [nan, nan, nan]},
                           columns=list('abfgh'))

        assert_frame_equal(df1, df2)

        df1 = 2 ** df
        df2 = dx.DataFrame({'a': [1, 1, 1024],
                            'b': [1, 4, nan],
                            'f': [1, 1024, 8],
                            'g': np.ones(3, dtype='int'),
                            'h': [nan, nan, nan]},
                           columns=list('abfgh'))
        assert_frame_equal(df1, df2)

    def test_mod_division_frame(self):
        with pytest.raises(TypeError):
            self.df % 5

        with pytest.raises(TypeError):
            self.df.select_dtypes('str') % 10

        with pytest.raises(TypeError):
            self.df % 'asdf'

        df = dx.DataFrame({'a': [6, 7, 10],
                           'b': [0, 2, nan],
                           'f': [0, 10, 3],
                           'g': np.zeros(3, dtype='int'),
                           'h': [nan, nan, nan]},
                          columns=list('abfgh'))

        df1 = df % 3
        df2 = dx.DataFrame({'a': [0, 1, 1],
                            'b': [0, 2, nan],
                            'f': [0, 1, 0],
                            'g': np.zeros(3, dtype='int'),
                            'h': [nan, nan, nan]},
                           columns=list('abfgh'))
        assert_frame_equal(df1, df2)

    def test_greater_than(self):
        with pytest.raises(TypeError):
            self.df > 5

        with pytest.raises(TypeError):
            self.df.select_dtypes('str') > 10

        with pytest.raises(TypeError):
            self.df > 'asdf'

        df = dx.DataFrame({'a': [6, 7, 10],
                           'b': [0, 2, nan],
                           'f': [0, 10, 3],
                           'g': np.zeros(3, dtype='int'),
                           'h': [nan, nan, nan]},
                          columns=list('abfgh'))

        df1 = df > 3
        df2 = dx.DataFrame({'a': [True, True, True],
                            'b': [False, False, False],
                            'f': [False, True, False],
                            'g': np.zeros(3, dtype='bool'),
                            'h': [False] * 3},
                           columns=list('abfgh'))
        assert_frame_equal(df1, df2)

    def test_greater_than_equal(self):
        with pytest.raises(TypeError):
            self.df >= 5

        with pytest.raises(TypeError):
            self.df.select_dtypes('str') >= 10

        with pytest.raises(TypeError):
            self.df >= 'asdf'

        df = dx.DataFrame({'a': [6, 7, 10],
                           'b': [0, 2, nan],
                           'f': [0, 10, 3],
                           'g': np.zeros(3, dtype='int'),
                           'h': [nan, nan, nan]},
                          columns=list('abfgh'))

        df1 = df >= 3
        df2 = dx.DataFrame({'a': [True, True, True],
                            'b': [False, False, False],
                            'f': [False, True, True],
                            'g': np.zeros(3, dtype='bool'),
                            'h': [False] * 3},
                           columns=list('abfgh'))
        assert_frame_equal(df1, df2)

    def test_less_than(self):
        with pytest.raises(TypeError):
            self.df < 5

        with pytest.raises(TypeError):
            self.df.select_dtypes('str') < 10

        with pytest.raises(TypeError):
            self.df < 'asdf'

        df = dx.DataFrame({'a': [6, 7, 10],
                           'b': [0, 2, nan],
                           'f': [0, 10, 3],
                           'g': np.zeros(3, dtype='int'),
                           'h': [nan, nan, nan]},
                          columns=list('abfgh'))

        df1 = df < 3
        df2 = dx.DataFrame({'a': [False, False, False],
                            'b': [True, True, False],
                            'f': [True, False, False],
                            'g': np.ones(3, dtype='bool'),
                            'h': [False] * 3},
                           columns=list('abfgh'))
        assert_frame_equal(df1, df2)

    def test_less_than_equal(self):
        with pytest.raises(TypeError):
            self.df <= 5

        with pytest.raises(TypeError):
            self.df.select_dtypes('str') <= 10

        with pytest.raises(TypeError):
            self.df <= 'asdf'

        df = dx.DataFrame({'a': [6, 7, 10],
                           'b': [0, 2, nan],
                           'f': [0, 10, 3],
                           'g': np.zeros(3, dtype='int'),
                           'h': [nan, nan, nan]})
        df1 = df <= 3

        df2 = dx.DataFrame({'a': [False, False, False],
                            'b': [True, True, False],
                            'f': [True, False, True],
                            'g': np.ones(3, dtype='bool'),
                            'h': [False] * 3
                            })
        assert_frame_equal(df1, df2)

    def test_neg_frame(self):
        with pytest.raises(TypeError):
            -self.df

        with pytest.raises(TypeError):
            -self.df.select_dtypes('str')

        df = dx.DataFrame({'a': [6, 7, 10],
                           'b': [0, 2, nan],
                           'f': [0, 10, 3],
                           'g': np.zeros(3, dtype='int'),
                           'h': [nan, nan, nan]})
        df1 = -df

        df2 = dx.DataFrame({'a': [-6, -7, -10],
                            'b': [0, -2, nan],
                            'f': [0, -10, -3],
                            'g': np.zeros(3, dtype='int'),
                            'h': [nan, nan, nan]})
        assert_frame_equal(df1, df2)

    def test_inplace_operators(self):
        with pytest.raises(NotImplementedError):
            self.df += 5

        with pytest.raises(NotImplementedError):
            self.df -= 5

        with pytest.raises(NotImplementedError):
            self.df *= 5

        with pytest.raises(NotImplementedError):
            self.df /= 5

        with pytest.raises(NotImplementedError):
            self.df //= 5

        with pytest.raises(NotImplementedError):
            self.df **= 5

        with pytest.raises(NotImplementedError):
            self.df %= 5


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
