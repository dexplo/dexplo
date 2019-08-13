import dexplo as dx
import numpy as np
from numpy import array, nan
import pytest
from dexplo.testing import assert_frame_equal, assert_array_equal, assert_dict_list


class TestSetItem:
    df = dx.DataFrame({'a': [1, 5], 'b': ['eleni', 'teddy'], 'c': [nan, 5.4],
                       'd': [True, False]})

    df1 = dx.DataFrame({'a': [1, 5, 7, 11], 'b': ['eleni', 'teddy', 'niko', 'penny'],
                        'c': [nan, 5.4, -1.1, .045], 'd': [True, False, False, True]})

    def test_setitem_scalar(self):
        df1 = self.df.copy()
        df1[0, 0] = -99
        df2 = dx.DataFrame({'a': [-99, 5], 'b': ['eleni', 'teddy'], 'c': [nan, 5.4],
                            'd': [True, False]})
        assert_frame_equal(df1, df2)

        df1[0, 'b'] = 'pen'
        df2 = dx.DataFrame({'a': [-99, 5], 'b': ['pen', 'teddy'], 'c': [nan, 5.4],
                            'd': [True, False]})
        assert_frame_equal(df1, df2)

        df1[1, 'b'] = None
        df2 = dx.DataFrame({'a': [-99, 5], 'b': ['pen', None], 'c': [nan, 5.4],
                            'd': [True, False]})
        assert_frame_equal(df1, df2)

        with pytest.raises(TypeError):
            df1 = self.df.copy()
            df1[0, 0] = 'sfa'

        df1 = self.df.copy()
        df1[0, 'c'] = 4.3
        df2 = dx.DataFrame({'a': [1, 5], 'b': ['eleni', 'teddy'], 'c': [4.3, 5.4],
                            'd': [True, False]})
        assert_frame_equal(df1, df2)

        df1 = self.df.copy()
        df1[0, 'a'] = nan
        df2 = dx.DataFrame({'a': [nan, 5], 'b': ['eleni', 'teddy'], 'c': [nan, 5.4],
                            'd': [True, False]})
        assert_frame_equal(df1, df2)

        df1 = self.df.copy()
        df1[1, 'a'] = -9.9
        df2 = dx.DataFrame({'a': [1, -9.9], 'b': ['eleni', 'teddy'], 'c': [nan, 5.4],
                            'd': [True, False]})
        assert_frame_equal(df1, df2)

    def test_setitem_entire_column_one_value(self):
        df1 = self.df.copy()
        df1[:, 'e'] = 5
        df2 = dx.DataFrame({'a': [1, 5], 'b': ['eleni', 'teddy'], 'c': [nan, 5.4],
                            'd': [True, False], 'e': [5, 5]})
        assert_frame_equal(df1, df2)

        df1 = self.df.copy()
        df1[:, 'e'] = nan
        df2 = dx.DataFrame({'a': [1, 5], 'b': ['eleni', 'teddy'], 'c': [nan, 5.4],
                            'd': [True, False], 'e': [nan, nan]})
        assert_frame_equal(df1, df2)

        df1 = self.df.copy()
        df1[:, 'e'] = 'grasshopper'
        df2 = dx.DataFrame({'a': [1, 5], 'b': ['eleni', 'teddy'], 'c': [nan, 5.4],
                            'd': [True, False], 'e': ['grasshopper', 'grasshopper']})
        assert_frame_equal(df1, df2)

        df1 = self.df.copy()
        df1[:, 'e'] = True
        df2 = dx.DataFrame({'a': [1, 5], 'b': ['eleni', 'teddy'], 'c': [nan, 5.4],
                            'd': [True, False], 'e': [True, True]})
        assert_frame_equal(df1, df2)

    def test_setitem_entire_new_colunm_from_array(self):
        df1 = self.df.copy()
        df1[:, 'e'] = np.array([9, 99])
        df2 = dx.DataFrame({'a': [1, 5], 'b': ['eleni', 'teddy'], 'c': [nan, 5.4],
                            'd': [True, False], 'e': [9, 99]})
        assert_frame_equal(df1, df2)

        df1 = self.df.copy()
        df1[:, 'e'] = [9, np.nan]
        df2 = dx.DataFrame({'a': [1, 5], 'b': ['eleni', 'teddy'], 'c': [nan, 5.4],
                            'd': [True, False], 'e': [9, np.nan]})
        assert_frame_equal(df1, df2)

        df1 = self.df.copy()
        df1[:, 'e'] = np.array([True, False])
        df2 = dx.DataFrame({'a': [1, 5], 'b': ['eleni', 'teddy'], 'c': [nan, 5.4],
                            'd': [True, False], 'e': [True, False]})
        assert_frame_equal(df1, df2)

        df1 = self.df.copy()
        df1[:, 'e'] = np.array(['poop', nan], dtype='O')
        df2 = dx.DataFrame({'a': [1, 5], 'b': ['eleni', 'teddy'], 'c': [nan, 5.4],
                            'd': [True, False], 'e': ['poop', nan]})
        assert_frame_equal(df1, df2)

        df1 = self.df.copy()
        df1[:, 'e'] = np.array(['poop', 'pants'])
        df2 = dx.DataFrame({'a': [1, 5], 'b': ['eleni', 'teddy'], 'c': [nan, 5.4],
                            'd': [True, False], 'e': ['poop', 'pants']})
        assert_frame_equal(df1, df2)

        df1 = self.df.copy()
        df1[:, 'e'] = np.array([nan, nan])
        df2 = dx.DataFrame({'a': [1, 5], 'b': ['eleni', 'teddy'], 'c': [nan, 5.4],
                            'd': [True, False], 'e': [nan, nan]})
        assert_frame_equal(df1, df2)

    def test_setitem_entire_new_colunm_from_list(self):
        df1 = self.df.copy()
        df1[:, 'e'] = [9, 99]
        df2 = dx.DataFrame({'a': [1, 5], 'b': ['eleni', 'teddy'], 'c': [nan, 5.4],
                            'd': [True, False], 'e': [9, 99]})
        assert_frame_equal(df1, df2)

        df1 = self.df.copy()
        df1[:, 'e'] = [9, np.nan]
        df2 = dx.DataFrame({'a': [1, 5], 'b': ['eleni', 'teddy'], 'c': [nan, 5.4],
                            'd': [True, False], 'e': [9, np.nan]})
        assert_frame_equal(df1, df2)

        df1 = self.df.copy()
        df1[:, 'e'] = [True, False]
        df2 = dx.DataFrame({'a': [1, 5], 'b': ['eleni', 'teddy'], 'c': [nan, 5.4],
                            'd': [True, False], 'e': [True, False]})
        assert_frame_equal(df1, df2)

        df1 = self.df.copy()
        df1[:, 'e'] = ['poop', nan]
        df2 = dx.DataFrame({'a': [1, 5], 'b': ['eleni', 'teddy'], 'c': [nan, 5.4],
                            'd': [True, False], 'e': ['poop', nan]})
        assert_frame_equal(df1, df2)

        df1 = self.df.copy()
        df1[:, 'e'] = ['poop', 'pants']
        df2 = dx.DataFrame({'a': [1, 5], 'b': ['eleni', 'teddy'], 'c': [nan, 5.4],
                            'd': [True, False], 'e': ['poop', 'pants']})
        assert_frame_equal(df1, df2)

        df1 = self.df.copy()
        df1[:, 'e'] = [nan, nan]
        df2 = dx.DataFrame({'a': [1, 5], 'b': ['eleni', 'teddy'], 'c': [nan, 5.4],
                            'd': [True, False], 'e': [nan, nan]})
        assert_frame_equal(df1, df2)

    def test_setitem_entire_old_column_from_array(self):
        df1 = self.df.copy()
        df1[:, 'd'] = np.array([9, 99])
        df2 = dx.DataFrame({'a': [1, 5], 'b': ['eleni', 'teddy'], 'c': [nan, 5.4],
                            'd': [9, 99]})
        assert_frame_equal(df1, df2)

        df1 = self.df.copy()
        d = np.array([9, np.nan])
        df1[:, 'd'] = d
        df2 = dx.DataFrame({'a': [1, 5], 'b': ['eleni', 'teddy'], 'c': [nan, 5.4],
                            'd': d})
        assert_frame_equal(df1, df2)

        df1 = self.df.copy()
        df1[:, 'd'] = np.array([True, False])
        df2 = dx.DataFrame({'a': [1, 5], 'b': ['eleni', 'teddy'], 'c': [nan, 5.4],
                            'd': [True, False]})
        assert_frame_equal(df1, df2)

        df1 = self.df.copy()
        df1[:, 'd'] = np.array(['poop', nan], dtype='O')
        df2 = dx.DataFrame({'a': [1, 5], 'b': ['eleni', 'teddy'], 'c': [nan, 5.4],
                            'd': ['poop', nan]})
        assert_frame_equal(df1, df2)

        df1 = self.df.copy()
        df1[:, 'a'] = np.array(['poop', 'pants'], dtype='O')
        df2 = dx.DataFrame({'a': ['poop', 'pants'], 'b': ['eleni', 'teddy'], 'c': [nan, 5.4],
                            'd': [True, False]})
        assert_frame_equal(df1, df2)

        df1 = self.df.copy()
        df1[:, 'b'] = np.array([nan, nan])
        df2 = dx.DataFrame({'a': [1, 5], 'b': [nan, nan], 'c': [nan, 5.4],
                            'd': [True, False]})
        assert_frame_equal(df1, df2)

        df1 = self.df.copy()
        df1[:, 'c'] = np.array([False, False])
        df2 = dx.DataFrame({'a': [1, 5], 'b': ['eleni', 'teddy'], 'c': [False, False],
                            'd': [True, False]})
        assert_frame_equal(df1, df2)

        with pytest.raises(ValueError):
            df1[:, 'b'] = np.array([1, 2, 3])

        with pytest.raises(ValueError):
            df1[:, 'b'] = np.array([1])

        with pytest.raises(TypeError):
            df1[:, 'a'] = np.array([5, {1, 2, 3}])

    def test_setitem_entire_new_column_from_df(self):
        df1 = self.df1.copy()
        df1[:, 'a_bool'] = df1[:, 'a'] > 3

        df2 = dx.DataFrame({'a': [1, 5, 7, 11], 'b': ['eleni', 'teddy', 'niko', 'penny'],
                            'c': [nan, 5.4, -1.1, .045], 'd': [True, False, False, True],
                            'a_bool': [False, True, True, True]},
                           columns=['a', 'b', 'c', 'd', 'a_bool'])
        assert_frame_equal(df1, df2)

        df1 = self.df1.copy()
        df1[:, 'a2'] = df1[:, 'a'] + 5

        df2 = dx.DataFrame({'a': [1, 5, 7, 11], 'b': ['eleni', 'teddy', 'niko', 'penny'],
                            'c': [nan, 5.4, -1.1, .045], 'd': [True, False, False, True],
                            'a2': [6, 10, 12, 16]},
                           columns=['a', 'b', 'c', 'd', 'a2'])
        assert_frame_equal(df1, df2)

    def test_setitem_entire_old_column_from_list(self):
        df1 = self.df.copy()
        df1[:, 'd'] = [9, 99]
        df2 = dx.DataFrame({'a': [1, 5], 'b': ['eleni', 'teddy'], 'c': [nan, 5.4],
                            'd': [9, 99]})
        assert_frame_equal(df1, df2)

        df1 = self.df.copy()
        df1[:, 'd'] = [9, np.nan]
        df2 = dx.DataFrame({'a': [1, 5], 'b': ['eleni', 'teddy'], 'c': [nan, 5.4],
                            'd': [9, np.nan]})
        assert_frame_equal(df1, df2)

        df1 = self.df.copy()
        df1[:, 'd'] = [True, False]
        df2 = dx.DataFrame({'a': [1, 5], 'b': ['eleni', 'teddy'], 'c': [nan, 5.4],
                            'd': [True, False]})
        assert_frame_equal(df1, df2)

        df1 = self.df.copy()
        df1[:, 'd'] = ['poop', nan]
        df2 = dx.DataFrame({'a': [1, 5], 'b': ['eleni', 'teddy'], 'c': [nan, 5.4],
                            'd': ['poop', nan]})
        assert_frame_equal(df1, df2)

        df1 = self.df.copy()
        df1[:, 'a'] = ['poop', 'pants']
        df2 = dx.DataFrame({'a': ['poop', 'pants'], 'b': ['eleni', 'teddy'], 'c': [nan, 5.4],
                            'd': [True, False]})
        assert_frame_equal(df1, df2)

        df1 = self.df.copy()
        df1[:, 'b'] = [nan, nan]
        df2 = dx.DataFrame({'a': [1, 5], 'b': [nan, nan], 'c': [nan, 5.4],
                            'd': [True, False]})
        assert_frame_equal(df1, df2)

        df1 = self.df.copy()
        df1[:, 'c'] = [False, False]
        df2 = dx.DataFrame({'a': [1, 5], 'b': ['eleni', 'teddy'], 'c': [False, False],
                            'd': [True, False]})
        assert_frame_equal(df1, df2)

        with pytest.raises(ValueError):
            self.df[:, 'b'] = [1, 2, 3]

        with pytest.raises(ValueError):
            self.df[:, 'b'] = [1]

        with pytest.raises(TypeError):
            self.df[:, 'a'] = [5, {1, 2, 3}]

    def test_setitem_simultaneous_row_and_column(self):
        df1 = self.df1.copy()
        df1[[0, 1], 'a'] = [9, 10]
        df2 = dx.DataFrame({'a': [9, 10, 7, 11], 'b': ['eleni', 'teddy', 'niko', 'penny'],
                            'c': [nan, 5.4, -1.1, .045], 'd': [True, False, False, True]})
        assert_frame_equal(df1, df2)

        df1 = self.df1.copy()
        df1[[0, -1], 'a'] = np.array([9, 10.5])
        df2 = dx.DataFrame({'a': [9, 5, 7, 10.5], 'b': ['eleni', 'teddy', 'niko', 'penny'],
                            'c': [nan, 5.4, -1.1, .045], 'd': [True, False, False, True]})
        assert_frame_equal(df1, df2)

        df1 = self.df1.copy()
        df1[2:, 'b'] = np.array(['NIKO', 'PENNY'])
        df2 = dx.DataFrame({'a': [1, 5, 7, 11], 'b': ['eleni', 'teddy', 'NIKO', 'PENNY'],
                            'c': [nan, 5.4, -1.1, .045], 'd': [True, False, False, True]})
        assert_frame_equal(df1, df2)

        df1 = self.df1.copy()
        df1[2, ['b', 'c']] = ['NIKO', 9.3]
        df2 = dx.DataFrame({'a': [1, 5, 7, 11], 'b': ['eleni', 'teddy', 'NIKO', 'penny'],
                            'c': [nan, 5.4, 9.3, .045], 'd': [True, False, False, True]})
        assert_frame_equal(df1, df2)

        df1 = self.df1.copy()
        df1[2, ['c', 'b']] = [9.3, None]
        df2 = dx.DataFrame({'a': [1, 5, 7, 11], 'b': ['eleni', 'teddy', None, 'penny'],
                            'c': [nan, 5.4, 9.3, .045], 'd': [True, False, False, True]})
        assert_frame_equal(df1, df2)

        df1 = self.df1.copy()
        df1[[1, -1], 'b':'d'] = [['TEDDY', nan, True], [nan, 5.5, False]]
        df2 = dx.DataFrame({'a': [1, 5, 7, 11], 'b': ['eleni', 'TEDDY', 'niko', nan],
                            'c': [nan, nan, -1.1, 5.5], 'd': [True, True, False, False]})
        assert_frame_equal(df1, df2)

        df1 = self.df1.copy()
        df1[1:-1, 'a':'d':2] = [[nan, 4], [3, 99]]

        df2 = dx.DataFrame({'a': [1, nan, 3, 11], 'b': ['eleni', 'teddy', 'niko', 'penny'],
                            'c': [nan, 4, 99, .045], 'd': [True, False, False, True]})
        assert_frame_equal(df1, df2)

    def test_testitem_boolean(self):
        df1 = self.df1.copy()
        criteria = df1[:, 'a'] > 4
        df1[criteria, 'b'] = 'TEDDY'
        df2 = dx.DataFrame({'a': [1, 5, 7, 11], 'b': ['eleni', 'TEDDY', 'TEDDY', 'TEDDY'],
                            'c': [nan, 5.4, -1.1, .045], 'd': [True, False, False, True]})
        assert_frame_equal(df1, df2)

        df1 = self.df1.copy()
        criteria = df1[:, 'a'] > 4
        df1[criteria, 'b'] = ['A', 'B', 'C']
        df2 = dx.DataFrame({'a': [1, 5, 7, 11], 'b': ['eleni', 'A', 'B', 'C'],
                            'c': [nan, 5.4, -1.1, .045], 'd': [True, False, False, True]})
        assert_frame_equal(df1, df2)

        df1 = self.df1.copy()
        criteria = df1[:, 'a'] == 5
        df1[criteria, :] = [nan, 'poop', 2.2, True]
        df2 = dx.DataFrame({'a': [1, nan, 7, 11], 'b': ['eleni', 'poop', 'niko', 'penny'],
                            'c': [nan, 2.2, -1.1, .045], 'd': [True, True, False, True]})
        assert_frame_equal(df1, df2)

        df1 = self.df1.copy()
        with pytest.raises(ValueError):
            df1[df1[:, 'a'] > 2, 'b'] = np.array(['aa', 'bb', 'cc', 'dd'])

        df1 = self.df1.copy()
        criteria = df1[:, 'a'] > 6
        df1[criteria, 'b'] = np.array(['food', nan], dtype='O')
        df2 = dx.DataFrame({'a': [1, 5, 7, 11], 'b': ['eleni', 'teddy', 'food', nan],
                            'c': [nan, 5.4, -1.1, .045], 'd': [True, False, False, True]})
        assert_frame_equal(df1, df2)

        df1 = self.df1.copy()
        df1[df1[:, 'a'] < 6, ['d', 'c', 'a']] = [[False, nan, 5.3], [False, 44, 4]]
        df2 = dx.DataFrame({'a': [5.3, 4, 7, 11], 'b': ['eleni', 'teddy', 'niko', 'penny'],
                            'c': [nan, 44, -1.1, .045], 'd': [False, False, False, True]})
        assert_frame_equal(df1, df2)

    def test_setitem_other_df(self):
        df_other = dx.DataFrame({'z': [1, 10, 9, 50], 'y': ['dont', 'be a', 'silly', 'sausage']})

        df1 = self.df1.copy()
        df1[:, ['a', 'b']] = df_other
        df2 = dx.DataFrame({'a': [1, 10, 9, 50], 'b': ['dont', 'be a', 'silly', 'sausage'],
                            'c': [nan, 5.4, -1.1, .045], 'd': [True, False, False, True]})
        assert_frame_equal(df1, df2)

        df1 = self.df1.copy()
        df1[[1, 3], ['c', 'b']] = df_other[[0, 2], :]
        df2 = dx.DataFrame({'a': [1, 5, 7, 11], 'b': ['eleni', 'dont', 'niko', 'silly'],
                            'c': [nan, 1, -1.1, 9], 'd': [True, False, False, True]})
        assert_frame_equal(df1, df2)

        with pytest.raises(ValueError):
            df1 = self.df1.copy()
            df1[[1, 3], ['c', 'b']] = df_other[[0], :]
