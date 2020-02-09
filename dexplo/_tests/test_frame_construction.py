import dexplo as dx
import numpy as np
from numpy import array, nan
import pytest
from dexplo.testing import assert_frame_equal, assert_array_equal, assert_dict_list


class TestFrameConstructorOneCol(object):

    def test_single_array_int(self):
        a = np.array([1, 2, 3])
        df1 = dx.DataFrame({'a': a})
        assert_array_equal(a, df1._data['i'][:, 0])
        assert df1._column_info['a'].values == ('i', 0, 0)

    def test_single_array_float(self):
        a = np.array([1, 2.5, 3.2])
        df1 = dx.DataFrame({'a': a})
        assert_array_equal(a, df1._data['f'][:, 0])
        assert df1._column_info['a'].values == ('f', 0, 0)

    def test_single_array_bool(self):
        a = np.array([True, False])
        df1 = dx.DataFrame({'a': a})
        assert_array_equal(a.astype('int8'), df1._data['b'][:, 0])
        assert df1._column_info['a'].values == ('b', 0, 0)

    def test_single_array_string(self):
        a = np.array(['a', 'b'])
        df1 = dx.DataFrame({'a': a})
        a1 = array([1, 2], dtype='uint32')
        assert_array_equal(a1, df1._data['S'][:, 0])
        assert df1._column_info['a'].values == ('S', 0, 0)

    def test_single_array_dt(self):
        a = np.array([10, 20, 30], dtype='datetime64[ns]')
        df1 = dx.DataFrame({'a': a})
        assert_array_equal(a, df1._data['M'][:, 0])
        assert df1._column_info['a'].values == ('M', 0, 0)

    def test_single_array_td(self):
        a = np.array([10, 20, 30], dtype='timedelta64[Y]')
        df1 = dx.DataFrame({'a': a})
        assert_array_equal(a.astype('timedelta64[ns]'), df1._data['m'][:, 0])
        assert df1._column_info['a'].values == ('m', 0, 0)

    def test_single_list_int(self):
        a = np.array([1, 2, 3])
        df1 = dx.DataFrame({'a': a.tolist()})
        assert_array_equal(a, df1._data['i'][:, 0])
        assert df1._column_info['a'].values == ('i', 0, 0)

    def test_single_list_float(self):
        a = np.array([1, 2.5, 3.2])
        df1 = dx.DataFrame({'a': a.tolist()})
        assert_array_equal(a, df1._data['f'][:, 0])
        assert df1._column_info['a'].values == ('f', 0, 0)

    def test_single_list_bool(self):
        a = np.array([True, False])
        df1 = dx.DataFrame({'a': a.tolist()})
        assert_array_equal(a.astype('int8'), df1._data['b'][:, 0])
        assert df1._column_info['a'].values == ('b', 0, 0)

    def test_single_list_string(self):
        a = np.array(['a', 'b'])
        df1 = dx.DataFrame({'a': a.tolist()})
        a1 = array([1, 2], dtype='uint32')
        assert_array_equal(a1, df1._data['S'][:, 0])
        assert df1._column_info['a'].values == ('S', 0, 0)

    def test_single_list_dt(self):
        a = [np.datetime64(x, 'ns') for x in [10, 20, 30]]
        df1 = dx.DataFrame({'a': a})
        assert_array_equal(np.array(a), df1._data['M'][:, 0])
        assert df1._column_info['a'].values == ('M', 0, 0)

    def test_single_list_td(self):
        a = [np.timedelta64(x, 'ns') for x in [10, 20, 30]]
        df1 = dx.DataFrame({'a': a})
        assert_array_equal(np.array(a), df1._data['m'][:, 0])
        assert df1._column_info['a'].values == ('m', 0, 0)


class TestFrameConstructorOneColArr(object):

    def test_single_array_int(self):
        a = np.array([1, 2, 3])
        df1 = dx.DataFrame(a)
        assert_array_equal(a, df1._data['i'][:, 0])
        assert df1._column_info['a0'].values == ('i', 0, 0)

    def test_single_array_float(self):
        a = np.array([1, 2.5, 3.2])
        df1 = dx.DataFrame(a)
        assert_array_equal(a, df1._data['f'][:, 0])
        assert df1._column_info['a0'].values == ('f', 0, 0)

    def test_single_array_bool(self):
        a = np.array([True, False])
        df1 = dx.DataFrame(a)
        assert_array_equal(a.astype('int8'), df1._data['b'][:, 0])
        assert df1._column_info['a0'].values == ('b', 0, 0)

    def test_single_array_string(self):
        a = np.array(['a', 'b'])
        df1 = dx.DataFrame(a)
        a1 = array([1, 2], dtype='uint32')
        assert_array_equal(a1, df1._data['S'][:, 0])
        assert df1._column_info['a0'].values == ('S', 0, 0)

    def test_single_array_dt(self):
        a = np.array([10, 20, 30], dtype='datetime64[ns]')
        df1 = dx.DataFrame(a)
        assert_array_equal(a, df1._data['M'][:, 0])
        assert df1._column_info['a0'].values == ('M', 0, 0)

    def test_single_array_td(self):
        a = np.array([10, 20, 30], dtype='timedelta64[Y]')
        df1 = dx.DataFrame(a)
        assert_array_equal(a.astype('timedelta64[ns]'), df1._data['m'][:, 0])
        assert df1._column_info['a0'].values == ('m', 0, 0)


class TestFrameConstructorMultipleCol(object):

    def test_array_int(self):
        a = np.array([1, 2, 3])
        b = np.array([10, 20, 30])
        arr = np.column_stack((a, b))
        df1 = dx.DataFrame({'a': a, 'b': b})
        assert_array_equal(arr, df1._data['i'])
        assert df1._column_info['a'].values == ('i', 0, 0)
        assert df1._column_info['b'].values == ('i', 1, 1)

    def test_array_float(self):
        a = np.array([1.1, 2, 3])
        b = np.array([10, 20.2, 30])
        arr = np.column_stack((a, b))
        df1 = dx.DataFrame({'a': a, 'b': b})
        assert_array_equal(arr, df1._data['f'])
        assert df1._column_info['a'].values == ('f', 0, 0)
        assert df1._column_info['b'].values == ('f', 1, 1)

    def test_array_bool(self):
        a = np.array([True, False, True])
        b = np.array([False, False, False])
        arr = np.column_stack((a, b)).astype('int8')
        df1 = dx.DataFrame({'a': a, 'b': b})
        assert_array_equal(arr, df1._data['b'])
        assert df1._column_info['a'].values == ('b', 0, 0)
        assert df1._column_info['b'].values == ('b', 1, 1)

    def test_array_string(self):
        a = np.array(['asdf', 'wer'])
        b = np.array(['wyw', 'xcvd'])
        df1 = dx.DataFrame({'a': a, 'b': b})
        a1 = array([[1, 1], [2, 2]], dtype='uint32')
        assert_array_equal(a1, df1._data['S'])
        assert df1._column_info['a'].values == ('S', 0, 0)
        assert df1._column_info['b'].values == ('S', 1, 1)

    def test_array_dt(self):
        a = np.array([10, 20, 30], dtype='datetime64[ns]')
        b = np.array([100, 200, 300], dtype='datetime64[ns]')
        arr = np.column_stack((a, b))
        df1 = dx.DataFrame({'a': a, 'b': b})
        assert_array_equal(arr, df1._data['M'])
        assert df1._column_info['a'].values == ('M', 0, 0)
        assert df1._column_info['b'].values == ('M', 1, 1)

    def test_array_td(self):
        a = np.array([10, 20, 30], dtype='timedelta64[Y]')
        b = np.array([1, 2, 3], dtype='timedelta64[Y]')
        arr = np.column_stack((a, b)).astype('timedelta64[ns]')
        df1 = dx.DataFrame({'a': a, 'b': b})
        assert_array_equal(arr, df1._data['m'])
        assert df1._column_info['a'].values == ('m', 0, 0)
        assert df1._column_info['b'].values == ('m', 1, 1)

    def test_array_int(self):
        a = np.array([1, 2])
        b = np.array([10, 20, 30])
        with pytest.raises(ValueError):
            dx.DataFrame({'a': a, 'b': b})


a = [1, 2, 5, 9, 3, 4, 5, 1]
b = [1.5, 8, 9, 1, 2, 3, 2, 8]
c = list('abcdefgh')
d = [True, False, True, False] * 2
e = [np.datetime64(x, 'D') for x in range(8)]
f = [np.timedelta64(x, 'D') for x in range(8)]
df_mix = dx.DataFrame({'a': a,
                       'b': b,
                       'c': c,
                       'd': d,
                       'e': e,
                       'f': f},
                      columns=list('abcdef'))


class TestAllDataTypesList:

    def test_all(self):
        assert_array_equal(np.array(a), df_mix._data['i'][:, 0])
        assert_array_equal(np.array(b), df_mix._data['f'][:, 0])
        a1 = array([1, 2, 3, 4, 5, 6, 7, 8], dtype='uint32')
        assert_array_equal(a1, df_mix._data['S'][:, 0])
        assert_array_equal(np.array(d).astype('int8'), df_mix._data['b'][:, 0])
        assert_array_equal(np.array(e, dtype='datetime64[ns]'), df_mix._data['M'][:, 0])
        assert_array_equal(np.array(f, dtype='timedelta64[ns]'), df_mix._data['m'][:, 0])

        assert df_mix._column_info['a'].values == ('i', 0, 0)
        assert df_mix._column_info['b'].values == ('f', 0, 1)
        assert df_mix._column_info['c'].values == ('S', 0, 2)
        assert df_mix._column_info['d'].values == ('b', 0, 3)
        assert df_mix._column_info['e'].values == ('M', 0, 4)
        assert df_mix._column_info['f'].values == ('m', 0, 5)


a1 = np.array([1, 2, 5, 9, 3, 4, 5, 1])
b1 = np.array([1.5, 8, 9, 1, 2, 3, 2, 8])
c1 = np.array(list('abcdefgh'), dtype='O')
d1 = np.array([True, False, True, False] * 2)
e1 = np.array(range(8), dtype='datetime64[D]')
f1 = np.array(range(8), dtype='timedelta64[D]')
df_mix1 = dx.DataFrame({'a': a,
                        'b': b,
                        'c': c,
                        'd': d,
                        'e': e,
                        'f': f},
                       columns=list('abcdef'))


class TestAllDataTypesArray:

    def test_all(self):
        assert_array_equal(a1, df_mix1._data['i'][:, 0])
        assert_array_equal(b1, df_mix1._data['f'][:, 0])
        arr1 = array([1, 2, 3, 4, 5, 6, 7, 8], dtype='uint32')
        assert_array_equal(arr1, df_mix1._data['S'][:, 0])
        assert_array_equal(d1.astype('int8'), df_mix1._data['b'][:, 0])
        assert_array_equal(e1, df_mix1._data['M'][:, 0])
        assert_array_equal(f1, df_mix1._data['m'][:, 0])

        assert df_mix1._column_info['a'].values == ('i', 0, 0)
        assert df_mix1._column_info['b'].values == ('f', 0, 1)
        assert df_mix1._column_info['c'].values == ('S', 0, 2)
        assert df_mix1._column_info['d'].values == ('b', 0, 3)
        assert df_mix1._column_info['e'].values == ('M', 0, 4)
        assert df_mix1._column_info['f'].values == ('m', 0, 5)


arr = np.column_stack((a1, b1, c1, d1, e1, f1))
df_mix2 = dx.DataFrame(arr)


class TestAllDataTypesObjectArray:

    def test_all(self):
        assert_array_equal(a1, df_mix2._data['i'][:, 0])
        assert_array_equal(b1, df_mix2._data['f'][:, 0])
        arr1 = array([1, 2, 3, 4, 5, 6, 7, 8], dtype='uint32')
        assert_array_equal(arr1, df_mix2._data['S'][:, 0])
        assert_array_equal(d1.astype('int8'), df_mix2._data['b'][:, 0])
        assert_array_equal(e1, df_mix2._data['M'][:, 0])
        assert_array_equal(f1, df_mix2._data['m'][:, 0])

        assert df_mix2._column_info['a0'].values == ('i', 0, 0)
        assert df_mix2._column_info['a1'].values == ('f', 0, 1)
        assert df_mix2._column_info['a2'].values == ('S', 0, 2)
        assert df_mix2._column_info['a3'].values == ('b', 0, 3)
        assert df_mix2._column_info['a4'].values == ('M', 0, 4)
        assert df_mix2._column_info['a5'].values == ('m', 0, 5)
