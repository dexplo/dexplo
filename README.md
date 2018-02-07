# dexplo
A data analysis library comparible to pandas

[![Build Status](https://travis-ci.org/dexplo/dexplo.svg?branch=master)](https://travis-ci.org/dexplo/dexplo)

# Main Goals
* A minimal set of features 
* Be as explicit as possible
* There should be one-- and preferably only one --obvious way to do it.

### Data Structures
* Only DataFrames
* No Series

### Data Types
* Only primitive types - int, float, boolean, str (unicode)
* No object data types

### Column Labels
* No hierarchical index
* Column names must be strings
* Column names must be unique
* Columns stored in a numpy array

### Row Labels
* No row labels for now
* Only a number display on the output
* Might add row labels in future

### Subset Selection
* Only one way to select data - `[ ]`
* Subset selection will be explicit and necessitate both rows and columns
* Rows will be selected only by integer location (for now)
* Columns will be selected by either label or integer location. Since columns must be strings, this will not be amibguous
* Column names cannot be duplicated
* Slice notation is also OK

### All selections and operations copy
* All selections and operations provide new copies of the data
* This will avoid any chained indexing confusion

### Development
* Must use type hints
* Must use 3.6 - fstrings
* Must have numpy


### Small feature set
* Implement as few attributes and methods as possible
* Focus on good idiomatic cookbook examples for doing more complex tasks

### Only Scalar Data Types
No complex Python data types
- [x] bool - always 8 bits, not-null
- [x] int - always 64 bits, not-null
- [x] float - always 64 bits, nulls allowed
- [x] str - A python unicode object, nulls allowed
- [ ] categorical
- [ ] datetime
- [ ] timedelta

#### Attributes to implement
- [x] size
- [x] shape
- [x] values
- [x] dtypes

May not implement any of the binary operators as methods (add, sub, mul, etc...)

#### Methods
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
- [ ] cumprod
- [x] cumsum
- [x] describe
- [x] max
- [x] min
- [x] median
- [x] mean
- [ ] mode
- [ ] nlargest
- [ ] nsmallest
- [x] quantile
- [ ] rank
- [x] std
- [x] sum
- [x] var
- [x] unique
- [x] nunique

**Selection**

- [ ] drop
- [ ] drop_duplicates
- [x] head
- [ ] isin
- [ ] sample
- [x] select_dtypes
- [x] tail
- [ ] where

**Missing Data**

- [x] isna
- [x] dropna
- [ ] fillna
- [ ] interpolate

**Group By** - specifically with `groupby` method
- [ ] agg
- [x] all
- [ ] apply
- [x] any
- [x] corr
- [x] count
- [x] cov
- [x] cumcount
- [ ] cummax
- [ ] cummin
- [ ] cumsum
- [x] head
- [x] first
- [ ] fillna
- [ ] filter
- [x] last
- [x] max
- [x] median
- [x] min
- [x] ngroups
- [x] nunique
- [ ] quantile
- [ ] rank
- [ ] rolling
- [x] size
- [x] sum
- [x] tail
- [x] var

**Other**

- [ ] append
- [ ] apply
- [ ] assign
- [x] astype
- [ ] groupby
- [ ] info
- [ ] join
- [ ] melt
- [ ] memory_usage
- [ ] pivot
- [ ] plot
- [ ] replace
- [ ] rolling
- [ ] sort_values

**Functions**

- [ ] read_csv
- [ ] read_sql
- [ ] concat

