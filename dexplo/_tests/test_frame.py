import dexplo as dx
import numpy as np
from numpy import array, nan
import pytest
from dexplo.testing import assert_frame_equal, assert_array_equal, assert_dict_list


class TestColumns:
    df1 = dx.DataFrame({'a': [1, 5, 7, 11],
                        'b': ['eleni', 'teddy', 'niko', 'penny'],
                        'c': [nan, 5.4, -1.1, .045],
                        'd': [True, False, False, True]})

    def test_set_columns_attr(self):
        df1 = self.df1.copy()
        df1.columns = ['z', 'y', 'x', 'w']
        df2 = dx.DataFrame({'z': [1, 5, 7, 11],
                            'y': ['eleni', 'teddy', 'niko', 'penny'],
                            'x': [nan, 5.4, -1.1, .045],
                            'w': [True, False, False, True]},
                           columns=['z', 'y', 'x', 'w'])
        assert_frame_equal(df1, df2)

        with pytest.raises(ValueError):
            self.df1.columns = ['sdf', 'er']

        with pytest.raises(ValueError):
            self.df1.columns = ['sdf', 'er', 'ewr', 'sdf']

        with pytest.raises(TypeError):
            self.df1.columns = [1, 2, 3, 4]

    def test_get_columns(self):
        columns = self.df1.columns
        assert (columns == ['a', 'b', 'c', 'd'])


class TestValues:
    df1 = dx.DataFrame({'a': [1, 5, 7, 11], 'b': [nan, 5.4, -1.1, .045]})
    df2 = dx.DataFrame({'a': [1, 5, 7, 11],
                        'b': [nan, 5.4, -1.1, .045],
                        'c': ['ted', 'fred', 'ted', 'fred']})

    def test_get_values(self):
        values1 = self.df1.values
        values2 = np.array([[1, 5, 7, 11], [nan, 5.4, -1.1, .045]]).T
        assert_array_equal(values1, values2)

        a = np.random.rand(100, 5)
        df = dx.DataFrame(a)
        assert_array_equal(df.values, a)

        values1 = self.df2.values
        values2 = np.array([[1, 5, 7, 11],
                            [nan, 5.4, -1.1, .045],
                            ['ted', 'fred', 'ted', 'fred']], dtype='O').T
        assert_array_equal(values1, values2)

    def test_shape(self):
        shape = self.df1.shape
        assert shape == (4, 2)

        a = np.random.rand(100, 5)
        df = dx.DataFrame(a)
        assert df.shape == (100, 5)

    def test_size(self):
        assert (self.df1.size == 8)

        a = np.random.rand(100, 5)
        df = dx.DataFrame(a)
        assert df.size == 500

    def test_to_dict(self):
        d1 = self.df1.to_dict('array')
        d2 = {'a': np.array([1, 5, 7, 11]),
              'b': np.array([nan, 5.4, -1.1, .045])}
        for key, arr in d1.items():
            assert_array_equal(arr, d2[key])

        d1 = self.df1.to_dict('list')
        d2 = {'a': [1, 5, 7, 11],
              'b': [nan, 5.4, -1.1, .045]}
        assert_dict_list(d1, d2)

    def test_copy(self):
        df2 = self.df1.copy()
        assert_frame_equal(self.df1, df2)


class TestSelectDtypes:
    data = {'a': [0, 0, 5, 9, 3, 4, 5, 1],
            'b': [0, 1.512344353, 8, 9, np.nan, 3, 2, 8],
            'c': [''] + list('bgggzgh'),
            'd': [False, False, True, False] * 2,
            'e': [0, 20, 30, 4, 5, 6, 7, 8],
            'f': [0., 3, 3, 3, 11, 4, 5, 1],
            'g': ['', None, 'ad', 'effd', 'ef', None, 'ett', 'zzzz'],
            'h': [0, 4, 5, 6, 7, 8, 9, 0],
            'i': np.array([0, 7, 6, 5, 4, 3, 2, 11]),
            'j': np.zeros(8, dtype='int'),
            'k': np.ones(8) - 1,
            'l': [np.nan] * 8}

    df = dx.DataFrame(data, columns=list('abcdefghijkl'))

    def test_selectdtypes_ints(self):
        df1 = self.df.select_dtypes('int')
        df2 = dx.DataFrame({'a': [0, 0, 5, 9, 3, 4, 5, 1],
                            'e': [0, 20, 30, 4, 5, 6, 7, 8],
                            'h': [0, 4, 5, 6, 7, 8, 9, 0],
                            'i': np.array([0, 7, 6, 5, 4, 3, 2, 11]),
                            'j': np.zeros(8, dtype='int')},
                           columns=list('aehij'))

        assert_frame_equal(df1, df2)

    def test_selectdtypes_float(self):
        df1 = self.df.select_dtypes('float')
        df2 = dx.DataFrame({'b': [0, 1.512344353, 8, 9, np.nan, 3, 2, 8],
                            'f': [0., 3, 3, 3, 11, 4, 5, 1],
                            'k': np.ones(8) - 1,
                            'l': [np.nan] * 8},
                           columns=list('bfkl'))
        assert_frame_equal(df1, df2)

    def test_selectdtypes_bool(self):
        df1 = self.df.select_dtypes('bool')
        df2 = dx.DataFrame({'d': [False, False, True, False] * 2})
        assert_frame_equal(df1, df2)

    def test_selectdtypes_str(self):
        df1 = self.df.select_dtypes('str')
        df2 = dx.DataFrame({'c': [''] + list('bgggzgh'),
                            'g': ['', None, 'ad', 'effd', 'ef', None, 'ett', 'zzzz']},
                           columns=['c', 'g'])
        assert_frame_equal(df1, df2)

    def test_selectdtypes_number(self):
        df1 = self.df.select_dtypes('number')
        df2 = dx.DataFrame({'a': [0, 0, 5, 9, 3, 4, 5, 1],
                            'b': [0, 1.512344353, 8, 9, np.nan, 3, 2, 8],
                            'e': [0, 20, 30, 4, 5, 6, 7, 8],
                            'f': [0., 3, 3, 3, 11, 4, 5, 1],
                            'h': [0, 4, 5, 6, 7, 8, 9, 0],
                            'i': np.array([0, 7, 6, 5, 4, 3, 2, 11]),
                            'j': np.zeros(8, dtype='int'),
                            'k': np.ones(8) - 1,
                            'l': [np.nan] * 8},
                           columns=list('abefhijkl'))
        assert_frame_equal(df1, df2)

    def test_get_dtypes(self):
        df1 = self.df.dtypes
        df2 = dx.DataFrame({'Column Name': list('abcdefghijkl'),
                            'Data Type': ['int', 'float', 'str', 'bool', 'int', 'float',
                                          'str', 'int', 'int', 'int', 'float', 'float']},
                           columns=['Column Name', 'Data Type'])
        assert_frame_equal(df1, df2)

    def test_selectdtypes_multiple(self):
        df1 = self.df.select_dtypes(['bool', 'int'])
        df2 = dx.DataFrame({'a': [0, 0, 5, 9, 3, 4, 5, 1],
                            'd': [False, False, True, False] * 2,
                            'e': [0, 20, 30, 4, 5, 6, 7, 8],
                            'h': [0, 4, 5, 6, 7, 8, 9, 0],
                            'i': np.array([0, 7, 6, 5, 4, 3, 2, 11]),
                            'j': np.zeros(8, dtype='int')}, columns=list('adehij'))
        assert_frame_equal(df1, df2)

        df1 = self.df.select_dtypes(['float', 'str'])
        df2 = dx.DataFrame({'b': [0, 1.512344353, 8, 9, np.nan, 3, 2, 8],
                            'c': [''] + list('bgggzgh'),
                            'f': [0., 3, 3, 3, 11, 4, 5, 1],
                            'g': ['', None, 'ad', 'effd', 'ef', None, 'ett', 'zzzz'],
                            'k': np.ones(8) - 1,
                            'l': [np.nan] * 8},
                           columns=list('bcfgkl'))
        assert_frame_equal(df1, df2)

        df1 = self.df.select_dtypes(exclude='float')
        df2 = dx.DataFrame({'a': [0, 0, 5, 9, 3, 4, 5, 1],
                            'c': [''] + list('bgggzgh'),
                            'd': [False, False, True, False] * 2,
                            'e': [0, 20, 30, 4, 5, 6, 7, 8],
                            'g': ['', None, 'ad', 'effd', 'ef', None, 'ett', 'zzzz'],
                            'h': [0, 4, 5, 6, 7, 8, 9, 0],
                            'i': np.array([0, 7, 6, 5, 4, 3, 2, 11]),
                            'j': np.zeros(8, dtype='int')})
        assert_frame_equal(df1, df2)
