import dexplo as dx
import numpy as np
from numpy import array, nan
import pytest
from dexplo.testing import assert_frame_equal


NaTdt = np.datetime64('nat')
NaTtd = np.timedelta64('nat')
df = dx.DataFrame({'a': [1, nan, 10, 0],
                   'b': ['a', 'a', 'c', 'c'],
                   'c': [5, 1, nan, 3],
                   'd': [True, False, True, nan],
                   'e': [3.2, nan, 1, 0],
                   'f': np.array([5, 10, NaTdt, 4], 'datetime64[Y]'),
                   'g': np.array([22, 10, NaTtd, 8], 'timedelta64[m]')
                   })


class TestAsTypeToBool:

    def test_bool_to_bool(self):
        assert_frame_equal(df, df)
        df1 = df.astype({'d': 'bool'})
        assert_frame_equal(df1, df)

        df1 = dx.DataFrame({'a': [True, False, True, nan],
                            'b': [True, False, True, True]})
        df2 = df1.astype('bool')
        assert_frame_equal(df1, df2)

    def test_int_to_bool(self):
        df1 = df.astype({'a': 'bool', 'c': 'bool'})
        df2 = dx.DataFrame({'a': [True, nan, True, False],
                            'b': ['a', 'a', 'c', 'c'],
                            'c': [True, True, nan, True],
                            'd': [True, False, True, nan],
                            'e': [3.2, nan, 1, 0],
                            'f': np.array([5, 10, NaTdt, 4], 'datetime64[Y]'),
                            'g': np.array([22, 10, NaTtd, 8], 'timedelta64[m]')
                            })
        assert_frame_equal(df1, df2)

        df1 = dx.DataFrame({'a': [1, 0, 10, nan],
                            'b': [5, 1, 14, nan]})
        df1 = df1.astype('bool')
        df2 = dx.DataFrame({'a': [True, False, True, nan],
                            'b': [True, True, True, nan]})
        assert_frame_equal(df1, df2)

    def test_float_to_bool(self):
        df1 = df.astype({'e': 'bool'})
        df2 = dx.DataFrame({'a': [1, nan, 10, 0],
                            'b': ['a', 'a', 'c', 'c'],
                            'c': [5, 1, nan, 3],
                            'd': [True, False, True, nan],
                            'e': [True, nan, True, False],
                            'f': np.array([5, 10, NaTdt, 4], 'datetime64[Y]'),
                            'g': np.array([22, 10, NaTtd, 8], 'timedelta64[m]')
                            })
        assert_frame_equal(df1, df2)

        df1 = dx.DataFrame({'a': [3.2, nan, 0],
                            'b': [0.0, 4, 2]})
        df1 = df1.astype('bool')
        df2 = dx.DataFrame({'a': [True, nan, False],
                            'b': [False, True, True]})
        assert_frame_equal(df1, df2)

    def test_str_to_bool(self):
        with pytest.raises(ValueError):
            df.astype({'b': 'bool'})

        df1 = dx.DataFrame({'a': ['a', 'a', 'c', 'c'],
                           'b': ['a', 'a', 'c', 'c']})
        with pytest.raises(ValueError):
            df1.astype('bool')

    def test_date_to_bool(self):
        pass

    def test_multiple_to_bool(self):
        pass


class TestAsTypeToInt:

    def test_bool_to_int(self):
        pass

    def test_int_to_int(self):
        pass

    def test_float_to_int(self):
        pass

    def test_str_to_int(self):
        pass

    def test_date_to_int(self):
        pass


class TestAsTypeToFloat:

    def test_bool_to_float(self):
        df1 = df.astype({'d': 'float'})
        df2 = dx.DataFrame({'a': [1, nan, 10, 0],
                            'b': ['a', 'a', 'c', 'c'],
                            'c': [5, 1, nan, 3],
                            'd': [1., 0, 1, nan],
                            'e': [3.2, nan, 1, 0],
                            'f': np.array([5, 10, NaTdt, 4], 'datetime64[Y]'),
                            'g': np.array([22, 10, NaTtd, 8], 'timedelta64[m]')
                            })
        assert_frame_equal(df1, df2)

        df1 = dx.DataFrame({'a': [True, False, True, True],
                            'b': [True, False, True, nan]})

        df1 = df1.astype('float')
        df2 = dx.DataFrame({'a': [1., 0, 1, 1],
                            'b': [1., 0, 1, nan]})
        assert_frame_equal(df1, df2)

    def test_int_to_float(self):
        df1 = df.astype({'a': 'float'})
        df2 = dx.DataFrame({'a': [1., nan, 10, 0],
                            'b': ['a', 'a', 'c', 'c'],
                            'c': [5, 1, nan, 3],
                            'd': [True, False, True, nan],
                            'e': [3.2, nan, 1, 0],
                            'f': np.array([5, 10, NaTdt, 4], 'datetime64[Y]'),
                            'g': np.array([22, 10, NaTtd, 8], 'timedelta64[m]')
                            })
        assert_frame_equal(df1, df2)

        df1 = df[:, ['a', 'c']].astype('float')
        df2 = dx.DataFrame({'a': [1., nan, 10, 0],
                            'c': [5., 1, nan, 3]})
        assert_frame_equal(df1, df2)

    def test_float_to_float(self):
        df1 = df.astype({'e': 'float'})
        df2 = dx.DataFrame({'a': [1, nan, 10, 0],
                            'b': ['a', 'a', 'c', 'c'],
                            'c': [5, 1, nan, 3],
                            'd': [True, False, True, nan],
                            'e': [3.2, nan, 1, 0],
                            'f': np.array([5, 10, NaTdt, 4], 'datetime64[Y]'),
                            'g': np.array([22, 10, NaTtd, 8], 'timedelta64[m]')
                            })
        assert_frame_equal(df1, df2)

    def test_str_to_foat(self):
        with pytest.raises(ValueError):
            df.astype({'b': 'float'})

        with pytest.raises(ValueError):
            df.astype('float')

    def test_date_to_float(self):
        df1 = dx.DataFrame({'a': np.array([5, 10], 'datetime64[Y]'),
                            'b': np.array([22, 10], 'timedelta64[m]')})
        with pytest.raises(ValueError):
            df1.astype({'a': 'float'})

        with pytest.raises(ValueError):
            df1.astype('float')


class TestAsTypeToStr:

    def test_bool_to_str(self):
        pass

    def test_int_to_str(self):
        df1 = df.astype({'a': 'str'})
        df2 = dx.DataFrame({'a': ['1', nan, '10', '0'],
                            'b': ['a', 'a', 'c', 'c'],
                            'c': [5, 1, nan, 3],
                            'd': [True, False, True, nan],
                            'e': [3.2, nan, 1, 0],
                            'f': np.array([5, 10, NaTdt, 4], 'datetime64[Y]'),
                            'g': np.array([22, 10, NaTtd, 8], 'timedelta64[m]')
                            })
        assert_frame_equal(df1, df2)

        df1 = df[:, ['a', 'c']].astype('str')
        df2 = dx.DataFrame({'a': ['1', nan, '10', '0'],
                            'c': ['5', '1', nan, '3']})
        assert_frame_equal(df1, df2)

    def test_float_to_str(self):
        pass

    def test_str_to_str(self):
        pass

    def test_date_to_str(self):
        pass


class TestAsTypeToDateTime:

    def test_bool_to_dt(self):
        pass

    def test_int_to_dt(self):
        pass

    def test_float_to_dt(self):
        pass

    def test_str_to_dt(self):
        pass

    def test_date_to_dt(self):
        pass


class TestAsTypeToTimeDelta:

    def test_bool_to_td(self):
        pass

    def test_int_to_td(self):
        pass

    def test_float_to_td(self):
        pass

    def test_str_to_td(self):
        pass

    def test_date_to_td(self):
        pass

