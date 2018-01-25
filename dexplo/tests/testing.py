def assert_frame_equal(df1, df2):
    for i, col in enumerate(df1.columns):
        if df2.columns[i] != col:
            raise AssertionError(f'column number {i} in left DataFrame not '
                                 f' {col} != {df2.columns[i]}')

    for i, (col1, col2) in enumerate(zip(df1.columns, df2.columns)):
        dt1 = df1._column_dtype[col1]
        dt2 = df2._column_dtype[col2]
        if dt1 != dt2:
            raise AssertionError(f'Data types for column {col1} does not '
                                 f'match {dt1} != {dt2}')
        idx1 = df1._dtype_column[dt1].index(col1)
        idx2 = df2._dtype_column[dt2].index(col2)

        data1 = df1._data[dt1][:, idx1]
        data2 = df2._data[dt2][:, idx2]

        if len(data1) != len(data2):
            raise AssertionError('DataFrames have different number of rows'
                                 f' {len(data1)} != {len(data2)}')

        if not (data1 == data2).all():
            bad_data1 = data1[data1 != data2][:5]
            bad_data2 = data2[data1 != data2][:5]
            bd = list(zip(bad_data1, bad_data2))
            raise AssertionError(f'Data is not equal {bd}')
