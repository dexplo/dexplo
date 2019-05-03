import dexplo as dx
import numpy as np
from numpy import nan
import pytest
from dexplo.testing import assert_frame_equal


class TestValueCounts:

    def test_value_counts(self):
        df = dx.DataFrame(
            {'AIRLINE': ['EV', 'VX', 'AA', 'UA', 'DL', 'B6', 'WN', 'AA', 'DL', 'AS', None, None],
             'DAY_OF_WEEK': [2, 1, 6, 4, 5, 5, 7, 5, 1, 4, 3, 3],
             'DEPARTURE_DELAY': [nan, -4.0, -1.0, -4.0, -1.0, 22.0, -3.0, 3.0, 21.0,
                                 -2.0, nan, 22]})
        df1 = df.value_counts('AIRLINE')
        df2 = dx.DataFrame({'AIRLINE': ['DL', 'AA', 'AS', 'WN', 'B6', 'UA', 'VX', 'EV'],
                            'count': [2, 2, 1, 1, 1, 1, 1, 1]})
        assert_frame_equal(df1, df2)

        df1 = df.value_counts('DAY_OF_WEEK')
        df2 = dx.DataFrame({'DAY_OF_WEEK': [5, 4, 3, 1, 7, 6, 2], 'count': [3, 2, 2, 2, 1, 1, 1]})
        assert_frame_equal(df1, df2)

        df1 = df.value_counts('DEPARTURE_DELAY')
        df2 = dx.DataFrame({'DEPARTURE_DELAY': [22.0, -1.0, -4.0, -2.0, 21.0, 3.0, -3.0],
                            'count': [2, 2, 2, 1, 1, 1, 1]})
        assert_frame_equal(df1, df2)

        df1 = df.value_counts('AIRLINE', normalize=True)
        df2 = dx.DataFrame({'AIRLINE': ['DL', 'AA', 'AS', 'WN', 'B6', 'UA', 'VX', 'EV'],
                            'count': [0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]})
        assert_frame_equal(df1, df2)

        df1 = df.value_counts('DEPARTURE_DELAY', normalize=True)
        df2 = dx.DataFrame({'DEPARTURE_DELAY': [22.0, -1.0, -4.0, -2.0, 21.0, 3.0, -3.0],
                            'count': [0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1]})
        assert_frame_equal(df1, df2)

        df1 = df.value_counts('DAY_OF_WEEK', normalize=True)
        df2 = dx.DataFrame({'DAY_OF_WEEK': [5, 4, 3, 1, 7, 6, 2],
                            'count': [0.25,
                                      0.16666666666666666,
                                      0.16666666666666666,
                                      0.16666666666666666,
                                      0.08333333333333333,
                                      0.08333333333333333,
                                      0.08333333333333333]})
        assert_frame_equal(df1, df2)

    def test_value_counts_sort_na(self):
        df = dx.DataFrame(
            {'AIRLINE': ['EV', 'VX', 'AA', 'UA', 'DL', 'B6', 'WN', 'AA', 'DL', 'AS', None, None],
             'DAY_OF_WEEK': [2, 1, 6, 4, 5, 5, 7, 5, 1, 4, 3, 3],
             'DEPARTURE_DELAY': [nan, -4.0, -1.0, -4.0, -1.0, 22.0, -3.0, 3.0, 21.0,
                                 -2.0, nan, 22]})

        df1 = df.value_counts('DAY_OF_WEEK', sort=False)
        df2 = dx.DataFrame({'DAY_OF_WEEK': [1, 2, 3, 4, 5, 6, 7], 'count': [2, 1, 2, 2, 3, 1, 1]})
        assert_frame_equal(df1, df2)

        df1 = df.value_counts('DEPARTURE_DELAY', sort=False)
        df2 = dx.DataFrame({'DEPARTURE_DELAY': [-4.0, -1.0, 22.0, -3.0, 3.0, 21.0, -2.0],
                            'count': [2, 2, 2, 1, 1, 1, 1]})
        assert_frame_equal(df1, df2)

        df1 = df.value_counts('DEPARTURE_DELAY', dropna=False)
        df2 = dx.DataFrame({'DEPARTURE_DELAY': [22.0, -1.0, -4.0, nan, -2.0, 21.0, 3.0, -3.0],
                            'count': [2, 2, 2, 2, 1, 1, 1, 1]})
        assert_frame_equal(df1, df2)

        df1 = df.value_counts('AIRLINE', dropna=False)
        df2 = dx.DataFrame({'AIRLINE': [None, 'DL', 'AA', 'AS', 'WN', 'B6', 'UA', 'VX', 'EV'],
                            'count': [2, 2, 2, 1, 1, 1, 1, 1, 1]})
        assert_frame_equal(df1, df2)

        df1 = df.value_counts('DEPARTURE_DELAY', dropna=False, sort=False)
        df2 = dx.DataFrame({'DEPARTURE_DELAY': [nan, -4.0, -1.0, 22.0, -3.0, 3.0, 21.0, -2.0],
                            'count': [2, 2, 2, 2, 1, 1, 1, 1]})
        assert_frame_equal(df1, df2)
