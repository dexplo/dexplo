dexplo
============

A data analysis library comparible to pandas

Main Goals
==========

-  A very minimal set of features
-  Be as explicit as possible
-  There should be one-- and preferably only one --obvious way to do it.

Data Structures
~~~~~~~~~~~~~~~

-  Only DataFrames
-  No Series

Data Types
~~~~~~~~~~

-  Only primitive types - int, float, boolean, numpy.unicode
-  No object data types

Row and Column Labels
~~~~~~~~~~~~~~~~~~~~~

-  No index, meaning no row labels
-  No hierarchical index
-  Column names must be strings
-  Column names must be unique
-  Columns stored in a numpy array

Subset Selection
~~~~~~~~~~~~~~~~

-  Only one way to select data - ``[ ]``
-  Subset selection will be explicit and necessitate both rows and
   columns
-  Rows will be selected only by integer location
-  Columns will be selected by either label or integer location. Since
   columns must be strings, this will not be amibguous
-  Column names cannot be duplicated

All selections and operations copy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  All selections and operations provide new copies of the data
-  This will avoid any chained indexing confusion

Development
~~~~~~~~~~~

-  Must use type hints
-  Must use 3.6 - fstrings
-  Must have numpy, bottleneck, numexpr

Small feature set
~~~~~~~~~~~~~~~~~

-  Implement as few attributes and methods as possible
-  Focus on good idiomatic cookbook examples for doing more complex
   tasks

Only Scalar Data Types
~~~~~~~~~~~~~~~~~~~~~~

No complex Python data types - [x] bool - always 8 bits, not-null - [x]
int - always 64 bits, not-null - [x] float - always 64 bits, nulls
allowed - [x] str - A python unicode object, nulls allowed - [ ]
categorical - [ ] datetime - [ ] timedelta

Attributes to implement
^^^^^^^^^^^^^^^^^^^^^^^

-  [x] size
-  [x] shape
-  [x] values
-  [x] dtypes

May not implement any of the binary operators as methods (add, sub, mul,
etc...)

Methods
^^^^^^^

**Stats** - [x] abs - [x] all - [x] any - [x] argmax - [x] argmin - [x]
clip - [ ] corr - [x] count - [ ] cov - [x] cummax - [x] cummin - [ ]
cumprod - [x] cumsum - [ ] describe - [x] max - [x] min - [x] median -
[x] mean - [ ] mode - [ ] nlargest - [ ] nsmallest - [ ] quantile - [ ]
rank - [x] std - [x] sum - [x] var - [ ] unique - [ ] nunique

**Selection** - [ ] drop - [ ] drop\_duplicates - [x] head - [ ] isin -
[ ] sample - [x] select\_dtypes - [x] tail - [ ] where

**Missing Data** - [ ] isna - [ ] dropna - [ ] fillna - [ ] interpolate

**Other** - [ ] append - [ ] apply - [ ] assign - [x] astype - [ ]
groupby - [ ] info - [ ] melt - [ ] memory\_usage - [ ] merge - [ ]
pivot - [ ] replace - [ ] rolling - [ ] sort\_values

**Functions** - [ ] read\_csv - [ ] read\_sql - [ ] concat
