# dexplo

[![](https://img.shields.io/pypi/v/dexplo)](https://pypi.org/project/dexplo)
[![Build Status](https://travis-ci.org/dexplo/dexplo.svg?branch=master)](https://travis-ci.org/dexplo/dexplo)
[![PyPI - License](https://img.shields.io/pypi/l/dexplo)](LICENSE)

A data analysis library comparable to pandas

## Installation

You must have cython installed. Run `python setup.py build_ext --use-cython -i`

## Main Goals

* A minimal set of features
* Be as explicit as possible
* There should be one-- and preferably only one --obvious way to do it.

### Data Structures

* Only DataFrames
* No Series

### Only Scalar Data Types

All data types allow nulls

- [x] bool - always 8 bits
- [x] int
- [x] float
- [x] str - stored as a categorical
- [x] datetime
- [x] timedelta

### Column Labels

* No hierarchical index
* Column names must be strings
* Column names must be unique

### Row Labels

* No row labels for now
* Only a number display on the output

### Subset Selection

* Only one way to select data - `[ ]`
* Subset selection will be explicit and necessitate both rows and columns
* Rows will be selected only by integer location
* Columns will be selected by either label or integer location. Since columns must be strings, this will not be amibguous
* Slice notation is also OK

### Development

* Must use type hints
* Must use 3.6+ - fstrings
* numpy

### Advantages over pandas

* Easier to write idiomatically
* String processing will be much faster
* Nulls allowed in each data type
* Nearly all operations will be faster

## API

### Attributes

- [x] size
- [x] shape
- [x] values
- [x] dtypes

### Methods

**Stats**

- [x] abs
- [x] all
- [x] any
- [x] argmax
- [x] argmin
- [x] clip
- [x] corr
- [x] count
- [x] cov
- [x] cummax
- [x] cummin
- [x] cumprod
- [x] cumsum
- [x] describe
- [x] max
- [x] min
- [x] median
- [x] mean
- [x] mode
- [x] nlargest
- [x] nsmallest
- [x] prod
- [x] quantile
- [x] rank
- [x] round
- [x] std
- [x] streak
- [x] sum
- [x] var
- [x] unique
- [x] nunique
- [x] value_counts

**Selection**

- [x] drop
- [x] head
- [x] isin
- [x] rename
- [x] sample
- [x] select_dtypes
- [x] tail
- [x] where

**Missing Data**

- [x] isna
- [x] dropna
- [x] fillna
- [ ] interpolate

**Other**

- [x] append
- [x] astype
- [x] factorize
- [x] groupby
- [x] iterrows
- [ ] join
- [x] melt
- [x] pivot
- [x] replace
- [x] rolling
- [x] sort_values
- [x] to_csv

**Other (after 0.1 release)**
- [ ] cut
- [ ] plot
- [ ] profile

**Functions**

- [x] read_csv
- [ ] read_sql
- [ ] concat

**Group By** - specifically with `groupby` method

- [x] agg
- [x] all
- [x] apply
- [x] any
- [x] corr
- [x] count
- [x] cov
- [x] cumcount
- [x] cummax
- [x] cummin
- [x] cumsum
- [x] cumprod
- [x] head
- [x] first
- [ ] fillna
- [x] filter
- [x] last
- [x] max
- [x] median
- [x] min
- [x] ngroups
- [x] nunique
- [x] prod
- [ ] quantile
- [ ] rank
- [ ] rolling
- [x] size
- [x] sum
- [x] tail
- [x] var

**str** - `df.str.<method>`

- [x] capitalize
- [x] cat
- [x] center
- [x] contains      
- [x] count         
- [x] endswith      
- [x] find         
- [x] findall
- [x] get           
- [x] get_dummies
- [x] isalnum
- [x] isalpha
- [x] isdecimal
- [x] isdigit
- [x] islower
- [x] isnumeric
- [x] isspace
- [x] istitle
- [x] isupper
- [x] join
- [x] len
- [x] ljust
- [x] lower         
- [x] lstrip
- [x] partition
- [x] repeat
- [x] replace
- [x] rfind
- [x] rjust
- [x] rpartition
- [x] rsplit
- [x] rstrip
- [x] slice
- [x] slice_replace
- [x] split
- [x] startswith
- [x] strip
- [x] swapcase
- [x] title
- [x] translate
- [x] upper         
- [x] wrap
- [x] zfill

**dt** - `df.dt.<method>`

- [x] ceil
- [x] day
- [x] day_of_week
- [x] day_of_year
- [x] days_in_month
- [x] floor
- [ ] freq
- [x] hour
- [x] is_leap_year
- [x] is_month_end
- [x] is_month_start
- [x] is_quarter_end
- [x] is_quarter_start
- [x] is_year_end
- [x] is_year_start
- [x] microsecond
- [x] millisecond
- [x] minute
- [x] month
- [x] nanosecond
- [x] quarter
- [x] round
- [x] second
- [x] strftime
- [x] to_pydatetime
- [x] to_pytime
- [ ] tz
- [ ] tz_convert
- [ ] tz_localize
- [x] weekday_name
- [x] week_of_year
- [x] year

**td** - `df.td.<method>`

- [ ] ceil
- [ ] components
- [x] days
- [ ] floor
- [ ] freq
- [x] microseconds
- [x] milliseconds
- [x] nanoseconds
- [ ] round
- [x] seconds
- [ ] to_pytimedelta