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
            self.df.select_dtypes('str') / some_bool

        with pytest.raises(TypeError):
            self.df / some_bool

        df1 = self.df.select_dtypes('number') / some_bool
        df2 = dx.DataFrame({'a': [0., nan, 5],
                            'b': [0, 1.5, nan],
                            'f': [0., 4, 5],
                            'g': np.zeros(3, dtype='float64'),
                            'h': [nan, nan, nan]},
                           columns=list('abfgh'))
        assert_frame_equal(df1, df2)

        df1 = some_bool / self.df.select_dtypes('number')
        df2 = dx.DataFrame({'a': [np.inf, nan, .2],
                            'b': [np.inf, 1 / 1.5, nan],
                            'f': [np.inf, .25, .2],
                            'g': [np.inf] * 3,
                            'h': [nan, nan, nan]},
                           columns=list('abfgh'))
        assert_frame_equal(df1, df2)

    def test_floordiv_int(self):
        with pytest.raises(TypeError):
            self.df // 5

        with pytest.raises(TypeError):
            self.df.select_dtypes('str') // 10

        with pytest.raises(TypeError):
            self.df // 'asdf'

        df = dx.DataFrame({'a': [0, 0, 10],
                           'b': [0, 20, nan],
                           'f': [0, 100, 10],
                           'g': np.zeros(3, dtype='int64'),
                           'h': [nan, nan, nan]},
                          columns=list('abfgh'))

        df1 = df // 3
        df2 = dx.DataFrame({'a': [0, 0, 3],
                            'b': [0, 6, nan],
                            'f': [0, 33, 3],
                            'g': np.zeros(3, dtype='int64'),
                            'h': [nan, nan, nan]},
                           columns=list('abfgh'))
        assert_frame_equal(df1, df2)

        df1 = df // 0
        df2 = dx.DataFrame({'a': [nan, nan, nan],
                            'b': [nan, nan, nan],
                            'f': [nan, nan, nan],
                            'g': [nan, nan, nan],
                            'h': [nan, nan, nan]},
                           columns=list('abfgh'))
        df2 = df2.astype('int').astype({'h': 'float'})
        assert_frame_equal(df1, df2)

    def test_floordiv_float(self):
        with pytest.raises(TypeError):
            self.df // 5.

        with pytest.raises(TypeError):
            self.df.select_dtypes('str') // 10.

        with pytest.raises(TypeError):
            self.df // 'asdf'

        df = dx.DataFrame({'a': [0., 0, 10],
                           'b': [0., 20, nan],
                           'f': [0., 100, 10],
                           'g': np.zeros(3, dtype='float64'),
                           'h': [nan, nan, nan]},
                          columns=list('abfgh'))

        df1 = df // 3.
        df2 = dx.DataFrame({'a': [0., 0, 3],
                            'b': [0., 6, nan],
                            'f': [0., 33, 3],
                            'g': np.zeros(3, dtype='float64'),
                            'h': [nan, nan, nan]},
                           columns=list('abfgh'))
        assert_frame_equal(df1, df2)

        df1 = df // 0.
        df2 = dx.DataFrame({'a': [nan, nan, nan],
                            'b': [nan, nan, nan],
                            'f': [nan, nan, nan],
                            'g': [nan, nan, nan],
                            'h': [nan, nan, nan]},
                           columns=list('abfgh'))
        assert_frame_equal(df1, df2)

    def test_floordiv_bool(self):
        some_bool = True
        with pytest.raises(TypeError):
            self.df // some_bool

        with pytest.raises(TypeError):
            self.df.select_dtypes('str') // some_bool

        df = dx.DataFrame({'a': [0, 0, 10],
                           'b': [0, 20, nan],
                           'f': [0, 100, 10],
                           'g': np.zeros(3, dtype='int'),
                           'h': [nan, nan, nan]},
                          columns=list('abfgh'))

        df1 = df // some_bool
        df2 = dx.DataFrame({'a': [0, 0, 10],
                            'b': [0, 20, nan],
                            'f': [0, 100, 10],
                            'g': np.zeros(3, dtype='int'),
                            'h': [nan, nan, nan]},
                           columns=list('abfgh'))
        assert_frame_equal(df1, df2)

        some_bool = False
        df1 = df // some_bool
        df2 = dx.DataFrame({'a': [nan, nan, nan],
                            'b': [nan, nan, nan],
                            'f': [nan, nan, nan],
                            'g': [nan, nan, nan],
                            'h': [nan, nan, nan]},
                           columns=list('abfgh'))
        df2 = df2.astype('int').astype({'h': 'float'})
        assert_frame_equal(df1, df2)

    def test_pow_int(self):
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

    def test_pow_float(self):
        with pytest.raises(TypeError):
            self.df ** 5.

        with pytest.raises(TypeError):
            self.df.select_dtypes('str') ** 10.

        with pytest.raises(TypeError):
            self.df ** 'asdf'

        df = dx.DataFrame({'a': [0, 0, 10],
                           'b': [0, 2, nan],
                           'f': [0, 10, 3],
                           'g': np.zeros(3, dtype='int64'),
                           'h': [nan, nan, nan]},
                          columns=list('abfgh'))
        df1 = df ** 2.
        df2 = dx.DataFrame({'a': [0., 0, 100],
                            'b': [0., 4, nan],
                            'f': [0., 100, 9],
                            'g': np.zeros(3, dtype='float64'),
                            'h': [nan, nan, nan]},
                           columns=list('abfgh'))

        assert_frame_equal(df1, df2)

        df1 = 2. ** df
        df2 = dx.DataFrame({'a': [1., 1, 1024],
                            'b': [1., 4, nan],
                            'f': [1., 1024, 8],
                            'g': np.ones(3, dtype='float64'),
                            'h': [nan, nan, nan]},
                           columns=list('abfgh'))
        assert_frame_equal(df1, df2)

    def test_pow_bool(self):
        some_bool = True    
        with pytest.raises(TypeError):
            self.df ** some_bool

        with pytest.raises(TypeError):
            self.df.select_dtypes('str') ** some_bool

        with pytest.raises(TypeError):
            self.df ** some_bool

        df = dx.DataFrame({'a': [0, 0, 10],
                           'b': [0, 2, nan],
                           'f': [0, 10, 3],
                           'g': np.zeros(3, dtype='int64'),
                           'h': [nan, nan, nan]},
                          columns=list('abfgh'))
        df1 = df ** some_bool
        df2 = dx.DataFrame({'a': [0, 0, 10],
                            'b': [0, 2, nan],
                            'f': [0, 10, 3],
                            'g': np.zeros(3, dtype='int64'),
                            'h': [nan, nan, nan]},
                           columns=list('abfgh'))

        assert_frame_equal(df1, df2)

        df1 = some_bool ** df
        df2 = dx.DataFrame({'a': [1, 1, 1],
                            'b': [1, 1, nan],
                            'f': [1, 1, 1],
                            'g': np.ones(3, dtype='int64'),
                            'h': [1., 1., 1.]},
                           columns=list('abfgh'))
        assert_frame_equal(df1, df2)

    def test_mod_int(self):
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

    def test_mod_float(self):
        with pytest.raises(TypeError):
            self.df % 5.

        with pytest.raises(TypeError):
            self.df.select_dtypes('str') % 10.

        with pytest.raises(TypeError):
            self.df % 'asdf'

        df = dx.DataFrame({'a': [6., 7, 10],
                           'b': [0., 2, nan],
                           'f': [0., 10, 3],
                           'g': np.zeros(3, dtype='float64'),
                           'h': [nan, nan, nan]},
                          columns=list('abfgh'))

        df1 = df % 3.
        df2 = dx.DataFrame({'a': [0., 1, 1],
                            'b': [0., 2, nan],
                            'f': [0., 1, 0],
                            'g': np.zeros(3, dtype='float64'),
                            'h': [nan, nan, nan]},
                           columns=list('abfgh'))
        assert_frame_equal(df1, df2)

    def test_mod_bool(self):
        some_bool = True
        with pytest.raises(TypeError):
            self.df % some_bool

        with pytest.raises(TypeError):
            self.df.select_dtypes('str') % some_bool

        with pytest.raises(TypeError):
            self.df % some_bool

        df = dx.DataFrame({'a': [6, 7, 10],
                           'b': [0, 2, nan],
                           'f': [0, 10, 3],
                           'g': np.zeros(3, dtype='int64'),
                           'h': [nan, nan, nan]},
                          columns=list('abfgh'))

        df1 = df % some_bool
        df2 = dx.DataFrame({'a': [0, 0, 0],
                            'b': [0, 0, nan],
                            'f': [0, 0, 0],
                            'g': np.zeros(3, dtype='int64'),
                            'h': [nan, nan, nan]},
                           columns=list('abfgh'))
        assert_frame_equal(df1, df2)

    def test_gt_int(self):
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
                            'b': [False, False, nan],
                            'f': [False, True, False],
                            'g': [False, False, False],
                            'h': [nan, nan, nan]},
                           columns=list('abfgh'))
        df2 = df2.astype('bool')
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
