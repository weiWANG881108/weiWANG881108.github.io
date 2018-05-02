---
layout: post
title: Pandas+Numpy
description: Operation
categories:
    - ML
comments: true
permalink: 
---

# Part 1. Pandas

*   Data structure
*   Index, selection and Filtering
*   Drop
*   Arithmetic and Data Alignment
*   Function Application and Mapping
*   Sorting and Ranking
*   Descriptive and summary statistics
*   Handle Missing Data and Replacing Values
*   Duplicates and Outliers
*   Discretization and Binning for continuous variables and Indicator/Dummy variable
*   Permutation and Random Sampling
*   String Manipulation
*   Combining and Merging Datasets
*   reshaping and Pivoting
*   GroupBy

## 1. Data structure
### (1) Series: one dimensional array-like object containing a sequence of values and an associated array of data labels, called <font color=red>Index</font>

| Name | Description | age         
| :- |-------------: | :-:
|Mary| She is a nice girl.  | 20
| Jackie Junior | He is a very naughty boy. | 5

* Initialization
  * Create from an array of data 
      * obj = pd.Series([4,7,-5,3])
      * obj = pd.Series([4,7,-5,3], <font color=red>index</font>=['d','b','c','a'])
  * Create from the dict
      * sdata = {'ohio': 3500, 'Texas': 71000, 'Oregon': 16000, 'Utah':5000}
* Get the array representation and index object of the Series
  * obj.<font color=red>values</font>
  * obj.<font color=red>index</font>
* Both Series and index object have a *name* attribute
  * obj.name
  * obj.index.name
* A series index can be altered in-place by assignment
  * obj.index = ['Bob', 'Steve', 'Jeff', 'Ryan']
* Operations will preserve the index-value link
  * obj*2
  * np.exp(obj)
  * obj.isnull()
* Others
  * 'b' in obj

### (2) DataFrame: a rectangular table of data and contains an ordered collection of columns, each of which can be a different value type.

* Initialization
  * a dict of equal-length list or numpy arrays, <font color=red>columns are placed in sortede order</font>
      * data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada', 'Nevada'],
                'year': [2000, 2001, 2002,2001,2002,2003], 
                'pop': [1.5, 1.7, 3.6, 2.4, 2.9, 3.2]} 
        frame = pd.DataFrame(data)
      * frame2 = pd.DataFrame(data, <font color=red>columns</font>=['year','state','pop','debt'], <font color=red>index</font>=['one','two','three','four','five','six'])
  * a nested dict of dict
      * pop = {'Nevada': {2001: 2.4, 2002:2.9}, 'ohio': {2000: 1.5,2001:1.7, 2002:3.6} }
  * dicts of series
* Two-dimensional ndarray and index, columns object of the Series
  * frame.<font color=red>index</font>
  * frame.<font color=red>columns</font>
  * frame.<font color=red>values</font>
* Both columns and index have a *name* attribute
  * obj.index.name
  * obj.columns.name
  * frame.head()
* Add a new column and delete a column
  * frame['eastern'] = frame.state=='Ohio'
  * delete frame['eastern']
* Index object
  * Initialization. labels = pd.Index(np.arange(3))
  * Index objects are immutable and can not be modified.
  * 'Ohio' in frame.columns
  * 2003 in frame.index
  * Pandas Index can contain duplicated labels.
  
| Method | Description          
| :- |-------------: 
|append| Concatenate with additional index objects, producing a new index 
| difference | compute set difference as an index
|intersection| compute set intersection
| union | compute set union
|isin| Compute boolean array indicating whether each value is contained in the passed collection
| delete | compute new index with element at index i deleted
|drop| compute new index by deleting passed value
| insert | compute new index by inserting element at element i
|is_monotonic| return true if each element is greater than or equal to the previous element
| is_unique | return true if the index has no duplicated values
| unique | Compute the array of unique values in the Index


## 2. Index, Selection and Filtering

### (1) Series

*  obj = pd.Series(np.arange(4.), index=['a','b','c','d'])
*  obj['b']
*  obj[['b','a','d']]
*  obj['b':'c'] or <font color=red>setting values</font> obj['b':'c']=5 <font color=red>end-point is inclusive</font>
*  obj[1]
*  obj[2:4]
*  obj[[1,3]]
*  obj[obj<2]

### (2) DataFrame

| Type | Notes          
| :- |-------------: 
|df[val]| Select single column or sequence of columns from the DataFrame 
| df.loc[val] | Select single row or subset of rows from the DataFrame by label
|df.loc[:,val]| Select single column or subset of columns by label
| df.loc[val1, val2] | Select both rows and columns by label
|df.iloc[where]| Select single row or subset of rows from the DataFrame by integer position
| df.iloc[:,where] | Select single column or subset of columns bu integer position
|df.iloc[where_i, where_j]| Select both rows and columns by integer position
| df.at[label_i, label_j] | Select a single scalar value by row and column label
|df.iat[i,j]| Select a single scalar value by row and column position (integers)
| reindex method | Select either rows or columns by labels
| get_value, set_value methods | Selet single value by row and column label

*  data = pd.DataFrame(np.arange(16).reshape(4,4), index=['Ohio','Colorado','Utah','New York'], columns=['one','two','three','four'])
*  data['two']
*  data[['three','one']]
*  data[:2]
*  data[data['three']>5]
*  data[data<5]=0

*  data.loc['Colorado',['two', 'three']]
*  data.iloc[2,[3,0,1]]
*  data.iloc[2]
*  data.iloc[[1,2],[2,0,1]]
*  data.loc[:'Utah','two']
*  data.iloc[:,:3][data.three>5]

### (3) Integer Indexes
*  ser = pd.Series(np.arange(3.0))
*  ser2 = pd.Series(np.arange(3.0), index=['a','b','c'])

*  ser[-1] // wrong
*  ser2[-1] // right

*  If you have an axis index containing integers, data selection will always be label-oriented. For more precise handling, use loc(for labels) or iloc(for integers):
    *  ser[:1]
    *  ser.loc[:1]
    *  ser.iloc[:1]
    
    
### (4) Reindex
*  obj = pd.Series([4.5,7.2,-5.3,3.6], index=['d','b','a','c'])
*  obj2 = obj.reindex(['a','b','c','d','e'])
*  obj3 = pd.Series(['blue','purple','yellow'], index=[0,2,4])
*  obj3.reindex(range(6),method='ffill')

### (5) Axis indexes with Duplicate Labels
*  obj = pd.Series(range(5), index=['a','a','b','b','c'])
    *  obj['a']  // return a series
    *  obj['c']  // return a scalar value
    *  obj.index.is_unique  --- is_unique property
*  df = pd.DataFrame(np.random.randn(4,3), index=['a','a','b','b'])
    *  df.loc['a']  // return a dataFrame

### (6) Renaming Axis Indexes
*  data = pd.DataFrame(np.arange(12).reshape((3,4)), index=['Ohio','Colorado','New York'], columns=['one','two', 'three','four'])
*  data.rename(index=str.title, columns=str.upper)
*  data.rename(index={'Ohio':'Indiana'}, columns={'three':'peekaboo'})
*  data.rename(index={'Ohio':'Indiana'}, inplace=True)

### (7) HIerarchical Indexing
*  data = pd.Series(np.random.randn(9), index=[['a','a','a','b','b','c','c','d','d'],[1,2,3,1,3,1,2,2,3]] )
*  data.index  ==> MultiIndex(levels=[['a', 'b', 'c', 'd'], [1, 2, 3]],labels=[[0, 0, 0, 1, 1, 2, 2, 3, 3], [0, 1, 2, 0, 2, 0, 1, 1, 2]])
*  data['b']
*  data['b':'c']
*  data.loc[['b','d']]
*  data.loc[:,2]
*  data.unstack()
*  data.unstack().stack()
*  frame = pd.DataFrame(np.arange(12).reshape((4,3)),index=[['a','a','b','b'],[1,2,1,2]], columns=[['Ohio','Ohio','Colorado'],['Green','Red','Green']])
*  frame.index.names = ['key1','key2']
*  frame.columns.names = ['state', 'color']
*  frame['Ohio']
*  frame.swaplevel('key1','key2')  ==> level interchange

### (8) Indexing with a DataFrame's columns
use one or more columns from a DataFrame as the row index
move the row index into the DataFrame's columns
* frame = pd.DataFrame({'a': range(7), 'b': range(7,0,-1), 'c':['one','one','one','two','two','two','two'], 'd':[0,1,2,0,1,2,3] })
* frame2 = frame.set_index(['c','d'])
* frame2 = frame.set_index(['c','d'], drop=False)
* frame2.reset_index()

## 3. Drop

### Series
*  obj = pd.Series(np.arange(5.0), index=['a','b','c','d','e'])
*  obj.drop('c')
*  obj.drop(['d','c'])

### DataFrame
*  data=pd.DataFrame(np.arange(16).reshape((4,4)), index=['Ohio','Colorado','Utah','New York'], columns=['one','two','three','four']  )
*  data.drop(['Colorado','Ohio'])
*  data.drop('two',axis=1)
*  data.drop(['two','four'], axis='columns')
*  data.drop('c', inplace=True)



## 4. Arithmetic and Data Alignment

*  When you are adding together objects, if any index pairs are not the same, the respective index in the result will be the union of the index pairs.
*  The internal data alignment introduces missing valuesin the label locations that don't overlap

### Arithmetic methods with fill values
| Method | Description          
| :- |-------------: 
| add, radd | Methods for addition(+)
| sub, rsub | Methods for subtraction(-)
| div, rdiv | Methods for division(/)
| floordiv, rfloordiv | Methods for floor division (//)
| mul, rmul | Methods for multiplication(*)
| pow, rpow | Methods for exponentiation(**)

*  df1 = pd.DataFrame(np.arange(12.).reshape((3,4)), columns=list('abcd'))
*  df2 = pd.DataFrame(np.arange(20.).reshape((4,5)), columns=list('abcde'))
*  df1.add(df2,fill_value=0)
*  df1.reindex(columns=df2.columns, fill_value=0)

#### Operations between DataFrame and Series
*  frame = pd.DataFrame(np.arange(12).reshape((4,3)), columns=list('bde'), index=['Utah','Ohio','Texas','Oregon'])
*  series = frame.iloc[0]
*  frame - series // <font color=red>broadcasting over rows</font>: the substraction is performed once for each row.
*  series2 = frame['d']
*  frame.sub(series2, axis='index')

## 5. Function Application and Mapping

### (1) NumPy ufuncs (element-wise array methods) also works with pandas objects:
*  frame=pd.DataFrame(np.random.randn(4,3), columns=list('bde'), index=['Utah','Ohio','Texas','Oregon'])
*  np.abs(frame)

### <font color=red>apply()</font>. 
*  function f returns a scalar value
    *  f = lambda x: x.max()-x.min()
    *  frame.apply(f)  // function f is involked once per column
    *  frame.apply(f,axis='columns')  // function f is involked once per row
*  function f returns a Series with multiple values
    *  def f(x):
            return pd.Series([x.min(),x.max()],index=['min','max'])
    *  frame.apply(f)

### <font color=red>applymap()</font>. element-wise Python function
* format = lambda x : '%.2f' %x
* frame.applymap(format)
* frame['e'].map(format)

### (2) map() function
*  data = pd.DataFrame({ 'food': ['bacon', 'pulled pork', 'bacon', 'pastrami', 'corned beef', 'bacon', 'pastrami', 'honey ham', 'nova lox'],
                         'ounces': [4,3,12,6,7.5,8,3,5,6]})
*  meat_to_animal={'bacon':'pig', 'pulled pork':'pig', 'pastrami':'cow', 'corned beef':'cow', 'honey ham':'pig', 'nova lox':'salmon'}
*  data['animal'] = data['food'].map(meat_to_animal)
*  data['food'].map(lambda x: meat_to_animal[x])

## 6. Sorting and Ranking

### sort_index(): sort by row or column index
*  frame = pd.DataFrame(np.arange(8).reshape((2,4)), index=['three', 'one'], columns=['d','a','b','c'])
*  frame.sort_index()
*  frame.sort_index(axis=1)
*  frame.sort_index(axis=1,ascending=False)
*  frame = pd.DataFrame(np.arange(12).reshape((4,3)),index=[['a','a','b','b'],[1,2,1,2]], columns=[['Ohio','Ohio','Colorado'],['Green','Red','Green']])
*  frame.sort_index(level=1)

### sort_values(): sort by its values
*  obj = pd.Series([4,np.nan,7,np.nan,-3,2])
*  obj.sort_values()   // any missing values are sorted to the end of the Series by default
*  frame = pd.DataFrame({'b':[4,7,-3,2], 'a':[0,1,0,1]})
*  frame.sort_values(<font color=red>by</font>='b')
*  frame.sort_values(<font color=red>by</font>=['a','b'])

### rank
*  obj = pd.Series([4,2,0,4], index=['a','b','c','d'])
    以值从小到大来赋排名值：c:0(1) b:2(2) a:4(3) d:4(4)
*  obj.rank()
        * a    3.5  求平均值(4+3)/2
        * b    2.0
        * c    1.0
        * d    3.5

| Method | Description          
| :- |-------------: 
| average | default: assign the average rank to each entry in the equal group
| min | Use the minimum rank for the whole group
| max | Use the maximum rank for the whole group
| first | Assign ranks in the roder the values appear in the data
| dense | Like method 'min', but ranks always increase by 1 in between groups rather than the number of equal elements in a group



## 7.Descriptive and summary statistics

### lists of summary statitics and related methods

| Method | Description          
| :- |-------------: 
| count | number of non-NA values
| describe | Computer set of summary statistics for Series or each DataFrame column
| min,max | Compute minimum and maximum values
| argmin, argmax | Compute index locations at which minimum or maximum value obtained, respectively
| idxmin, idxmax | Compute index labels at which minimum or maximum value obtained, respectively
| sum | sum of values
| mean | mean of values
| median | Arithmetic median (50% quantile) of value
| quantile | Compute sample quantile ranging from 0 to 1
| mad | mean absolute deviation from mean value
| prod | Product of all values
| var | sample variance of values
| std | sample standard derivation of values
| skew | sample skewness of values
| kurt | sample kurtosis of values
| cumsum | cumulative sum of values
| cummin,cummax | Cumulative minimum or maximum of values, repectively
| cumprod | cumulative product of values
| diff | Compute first arithmetic difference (useful for time series)
| pct_change | Compute percentage change

*  On non-numeric data, describe() produces alternative summary statistics:
    *  obj = pd.Series(['a','a','b',c] * 4)
    *  obj.describe(). -->  count: 16, unique:3, top:a, freq: 8
*   frame = pd.Series(np.random.randn(9), index=[['a','a','a','b','b','c','c','d','d'],[1,2,3,1,3,1,2,2,3]] )
    *  frame.index.names = ['key1','key2']
    *  frame.columns.names = ['state', 'color']
    *  frame.sum(level='key2')
    *  frame.sum(level='color', axis=1)

### Correlation and Covariance
*  returns is a DataFrame. It has four columns ['AAPL', 'GOOG', 'IBM', 'MSFT']
*  returns['MSFT'].corr(returns['IBM']) or returns.MSFT.corr(returns.IBM)
*  returns['MSFT'].cov(returns['IBM'])
*  returns.corr()
*  corrwith() method.
    *  passing a series returns a Series with the correlation value computed for each column. returns.corrwit(returns.IBM)
    *  passing a DataFrame returns the correlations of matching column names.
    
### Unique Values, Value Counts and Membership
| Method | Description          
| :- |-------------: 
| isin | Compute boolean array indicating whether each Series value is contained in the passed sequence of values
| match | Compute integer indices for each value in an array into another array of distinct values; helpful for data alignment and join-type operations
| unique | Compute array of unique values in Series, returned in the order observed
| value_counts | Return a Series containing unique valeus as its index and frequencies as its value, ordered count in descending order

*  Index.get_indexer() method gives you an index array from an array of possibly non-distinct values into another array of distinct values;
    * to_match = pd.Series(['c','a','b','b','c','a'])
    * unique_vals = pd.Series(['c','b','a'])
    * pd.Index(unique_vals).get_indexer(to_match)
    * result--> array([0,2,1,1,0,2])
*  compute a hitogram on multiple related columns in a DataFrame
    * data = pd.DataFrame({'Qu1':[1,3,4,3,4], 'Qu2':[2,3,1,2,3], 'Qu3':[1,5,2,4,4]})
    * result = data.apply(pd.value_counts).fillna(0)

## 8. Handling Missing Data and Replacing Values

### 1. NA: not avaiable. Missing data.
*  np.nan
*  None: built-in Python value

### 2. NA handling methods

| Method | Description          
| :- |-------------: 
| dropna | Filter axis labels based on whether values for each label having missing data, with varying thresholds for how much missing data to tolerate
| fillna | Fill in missing data with some value or using an iterpolation method such as 'ffill' or 'bfill'
| isnull | Return boolean values indicating which values are missing/NA
| notnull | Negation of isnull



#### (1) dropna() function
*  from numpy import nan as NA
*  data = pd.Series([1,NA,3.5,NA,7])
*  data.dropna(). <==> data[data.notnull()]
*  data = pd.DataFrame([[1.,6.5,3.],[1.,NA,NA],[NA,NA,NA],[NA,6.5,3.]])
*  cleaned = data.dropna() ==> by default, drops any row containg a missing value
*  data.dropna(how='all')  ==> only drop rows that are all NA:
*  data.dropna(axis=1, how='all')  ==> drop columns that are all NA
*  data.dropna(thresh=2)  ==> keep rows containing a certain number of observations

#### (2)  fillna() function

fillin() function arguments

| Argument | Description          
| :- |-------------: 
| value | scalar value or dict-like object to use to fill missing values
| method | Interpolation; by default 'ffill' if function called with no other arguments
| axis | Axis to fill on; defulat axis = 0
| inplace | Modify the calling object with producing a copy
| limit | For forward and backward filling, maximum number of consecutive periods to fill

*  df.fillna(0) ==>fill constant for missing data
*  df.fillna(data.mean())
*  df.fillna({1:0.5, 2:0}) ==>fill a different value for each column
*  df.fillna(method='ffill', limit=2)

### 3. Replacing Values
*  data=pd.Series([1.,-999.,2.,-999.,-1000.,3.])
*  data.replace(-999, np.nan)
*  data.replace([-999,-1000], np.nan)
*  data.replace([-999.-1000],[np.nan,0])
*  data.replace({-999:np.nan,-1000:0})

## 9. Duplicates and Outliers

### DataFrame method duplicated() and drop_duplicates()

*  data = pd.DataFrame({'k1':['one','two']*3 + ['two'], 'k2':[1,1,2,3,3,4,4]})
*  data.duplicated()  ==> whether each row is duplcated
*  data.drop_duplicates(). ==> returns a DataFrame where the duplicated array is False
*  data.drop_duplicates(['k1'])  ==> only consider column 'k1'
*  data.drop_duplicates(['k1','k2'], keep='last') 

###  Outliers

* data = pd.DataFrame(np.random.randn(1000,4))
* data[(np.abs(data)>3).any(1)]  ==> select rows having a value exceeding 3 or -3
* data[np.abs(data)>3] = np.sign(data) * 3   // cap values outside the interval -3 and 3. np.sign(data): produces 1 and -1

## 10. Discretization and Binning for continuous variables and Indicator/Dummy variable

### (1) DataFrame method cut() and qcut() ==> Discretization and Binning
* ages=[20,22,25,27,21,23,37,31,61,45,41,32]
* bins = [18,25,35,60,100]  ==> divide these into bins of 18 to 25, 26 to 35, 36 to 60 and 61 to 100
* cats = pd.cut(ages, bins)  ==> [(18,25],(18,25],...,(25,35]]
* cats.codes ==> labels for ages data, array([0,0,0,1,0,0,2,1,3,2,2,1], dtype=int8)
* cats.categories  ==> IntervalIndex([(18,25], (25,35], (35,60], (60,100]])       
* pd.value_counts(cats)
* cats = pd.cut(ages, bins, right=False)
* group_name = [''Youth', 'YoungAdult','MiddleAge','Senior']
* pd.cut(ages, bins, labels=group_name)
* data=np.random.rand(20)
* pd.cut(data,4,precision=2) ==> compute equal-length bins. precision=2 limits the decimal precision to two digits
* data=np.random.randn(1000)
* cats = pd.qcut(data,4) ==> cut into quartiles
* pd.value_counts(cats)  ==> obtain roughly equal-size bins
* pd.qcut(data,[0,0.1,0.5,0.9,1])  ==>  pass your own quantiles

### (2) Indicator/Dummy variable

#### get_dummies() function
* df = pd.DataFrame({'key':['b','b','a','c','a','b'], 'data1': range(6)})
* pd.get_dummmies(df['key']) ==> transform a column into k columns containing all 1s and 0s
* dummies = pd.get_dummies(df['key'], prefix='key')
* df_with_dummy = df[['data1']].join(dummies)

If a row in a DataFrame belongs to multiple categories

*  Example MovieLens 1M dataset
*  mnames = ['movie_id', 'title', 'genres']
*  movies = pd.read_table('datasets/movielens/movies.dat', sep='::', header=None, names=mnames)

Add dummy variable for the genres columns

*  all_genres = []
*  for x in movies.genres:
        all_genres.extend(x.split('|'))
*  genres = pd.unique(all_genres)
*  zero_matrix = np.zeros( (len(movies), len(genres)) )
*  dummies = pd.DataFrame(zero_matrix, columns=genres)
*  for i, gen in enumerate(movies.genres):
        indices = dummies.columns.get_indexer(gen.split('|'))
        dummies.iloc[i,indices] = 1
*  movies_windic = movies.join(dummies.add\_prefix('Genre_'))

Combining get_dummies() and a discretization function like cut:
*  np.random.seed(12345)
*  values = np.random.rand(10)
*  bins = [0,0.2,0.4,0.6,0.8,1]
*  pd.get_dummies(pd.cut(values, bins))

## 11. Permutation and Random Sampling

### numpy.random.permutation
* df = pd.DataFrame(np.arangre(5*4).reshape((5,4)))
* sampler = np.random.permutation(5)
* df.take(sampler)
### sample() method on Series and DataFrame
* df.sample(n=3)
* df.sample(n=10, replace=True)   ==> sample with replacement

## 12. String Manipulation

### Python built-in string methods

| Argument | Description          
| :- |-------------: 
| count | Return the number of non-overlapping occurences of substring in the string
| endswith | Return True if string ends with suffix
| startswith | Return True if string starts with prefix
| join | Use string as delimiter for concatenating a sequence of other strings
| index | Return position of first character in substring if found in the string; raise ValueError if not found
| find | Return position of first character of first occurrence of substring in the string; like index, but returns -1 if not found
| rfind | Return position of first character of last occurrence of substring in the string; return -1 if not found
| replace | Replace occurences of string with another string
| strip | Trim whitespace, including newlines;
| rstrip | 
| lstrip | 
| split | Break string into list of substrings using passed delimiter
| lower | Convert alphabet characters to lowercase
| upper | Convert alphabet characters to uppercase
| casefold | Convert characters to lowercase, and convert any region-specific variable character combinations toa common comparable form
| ljust | Left justify. pad opposite side of string with spaces (or some other fill character) to return a string with minimum width
| rjust | 

### Regular Expressions

| Argument | Description          
| :- |-------------: 
| findall | Return all non-overlapping matching pattern in a string as a list
| finditer | like findall, but returns an iterator
| match | Match pattern at start of string and optionally segment pattern components into groups; if the pattern matches, return a match object, and otherwise None
| search | Scan string for match to pattern; returning a match object if so; unlike match, the match can be anywhere in the string as opposed to only at the beginning
| split | Break string into pieces at each occurrence of pattern
| sub,subn | Replace all (sub) or first n occurence (subn) of pattern in string with replacement expression; use symbols \1,\2,... to refer to match group elements in the replaement string

*  import re
*  text = "foo.  bar\t baz  \tqux"
*  re.split('\s+', text)
*  regex = re.compile('\s+')
*  regex.split(text)
*  regex.findall(text)

findall returns all matches in a string, search returns only the first match, match only matches at the beginning of the string

*  text="""
         Dave dave@google.com
         Steve steve@gmail.com
         Rob rob@gmail.com
         Ryan ryan@yahoo.com
     """
*  pattern = r'[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,4}'
*  regex = re.compile(pattern, flags=re.IGNORECASE)
*  regex.findall(text)
*  m = regex.search(pattern)
*  text[m.start(): m.end()]
*  print(regex.match(pattern))  ==> None
*  print(regex.sub('REDACTED',text))

find and segment email address, using parentheses
*  pattern = r'([A-Z0-9._%+-]+)@([A-Z0-9.-]+)\.([A-Z]{2,4})'
*  regex = re.compile(pattern, flags=re.IGNORECASE)
*  regex.match('wesm@bright.net')
*  m.groups()
*  regex.findall(text)
*  print(regex.sub(r'Username: \1, Domain: \2, Suffix: \3', text))

### Vectorized String Functions in pandas

| Argument | Description          
| :- |-------------: 
| cat | Concatenate strings elements-wise with optional delimiter
| contains | Return boolean array if each string contains pattern/regex
| count | Count occurences of pattern
| extract | Use a regular expression with groups to extract one or more strings from a Series of strings; the result will be a DataFrame with one column per group
| endswith | Equivalent to x.endswith(pattern) for each element
| startswith | Equivalent to x.startswith(pattern) for each element
| findall | Compute list of all occurrences of pattern/regex for each string
| get | Index into each element
| isalnum | Equivalent to built-in str.alnum
| isalpha | Equivalent to built-in str.isalpha
| isdecimal | Equivalent to built-in str.isdecimal
| isdigit | Equivalent to built-in str.isdigit
| islower | Equivalent to built-in str.islower
| isnumeric | Equivalent to built-in str.isnumeric
| isupper | Equivalent to built-in str.isupper
| join | Join strings in each element of the Series with passed separator
| len | Compute the length of each string
| lower, upper | Convert cases; equivalent to x.lower() or x.upper() for each element
| match | Use re.match with the passed regular expression on each element, returning matched group as list
| pad | Add whitespace to left, right, or both sides of strings
| center | Equivalent to pad (side='both')
| repeat | Duplicate values is equivalent to x*3 for each string
| replace | Replace occurrences of pattern/regex with some other string
| slice | Slice each string in the Series
| split | Split strings on delimiter or regular expression
| strip | Trim whitespace from both side, including newlines
| rstrip | Trim whitespace on right side
| lstrip | Trim whitespace on left side

map() function of DataFrame or Series can apply string and regular expression methods to each value, but it will fail on the NA values. To cope with this, Series has array-oriented methods for string operations that skip NA values. These are accessed throught Series's str attribute

* data = pd.Series(['a_b_c', 'c_d_e', np.nan, 'f_g_h'])
* data.str.split('_').str.get(1) or data.str.split('_').str[1]
* data = {'Dave': 'dave@google.com', 'Steve': 'steve@gmail.com', 'Rob': 'rob@gmail.com', 'Wes': np.nan}
* data = pd.Series(data)
* data.str.contains('gmail')
* pattern = r'([A-Z0-9._%+-]+)@([A-Z0-9.-]+)\.([A-Z]{2,4})'
* data.str.findall(pattern, flags = re.IGNORECASE)
* matches = data.str.match(pattern,flags=re.IGNORECASE)
* data.str[:5]

## 13. Combining and Merging Datasets

### (1)  ways to combine
* pandas.merge, join: connects rows in DataFrame based on one or more keys
* pandas.concat: concatenates or "stack" together objects along an axis
* combine_first: enables splicing together overlapping data to fill in missing values in one object with values from another

### (2) merge() and join() method

#### merge() function arguments:

| Argument | Description          
| :- |-------------: 
| left | DataFrame to be merged on the left side
| right | DataFrame to be merged on the right side 
| how | One of 'inner', 'outer', 'left' or 'right'; default to 'inner' 
| on | Column names to join on. Must be found in both DataFrame objects. If not specified and no other join keys given, will use the intersection of the columns in left and right as the join key
| left_on | Columns in left DataFrame to use as join keys 
| right_on | Analogous to left_on for left DataFrame 
| left_index | Use row index in lefts as join key (or keys)
| right_index | Analogous to left_index
| sort | Sort merged data lexicographically by join keys; true by default
| suffixes | Tuple of string values to append to column names in case of overlap; default to ('_x', '_y') 
| copy | if false, avoid copying data into resulting data structure in some exceptional cases; by default always copies
| indicator | Adds a special column _merge that indicates the source of each row; values will be 'left_only', 'right_only', or 'both' based on the origin of the joined data in each row

#### different join types with how argument

| Argument | Description          
| :- |-------------: 
| inner | Use only the key combinations obsered in both tables
| left | Use all key combinations found in the left table
| right | Use all key combinations found in the right table
| output | Use all key combinations observed in both tables together

##### Many-to-One join:
* df1=pd.DataFrame({'key':['b','b','a','c','a','a','b'],'data1':range(7)})
* df2 = pd.DataFrame({'key':['a','b','d'],'data2':range(3)})
* pd.merge(df1,df2) ==> the data in df1 has multiple rows labeled a and b, whereas df2 has only one row for each value in the key column
* pd.merge(df1,df2,on='key')
* df3=pd.DataFrame({'lkey':['b','b','a','c','a','a','b'],'data1':range(7)})
* df4=pd.DataFrame({'rkey':['a','b','d'],'data2':range(3)})
* pd.merge(df1,df2,left_on='lkey', right_on='rkey')
* pd.merge(df1,df2,how='outer')

##### Many-to-many
* df1=pd.DataFrame({'key':['b','b','a','c','a','b'],'data1':range(7)})
* df2 = pd.DataFrame({'key':['a','b','a','b','d'],'data2':range(5)})
* pd.merge(df1,df2,on='key',how='left')  ==> Cartesian product of the rows.
* pd.merge(df1,df2,how='inner')

##### to merge multiple keys
* left=pd.DataFrame({'key1':['foo','foo','bar'],'key2':['one','two','one'],'ival':[1,2,3]})
* right=pd.DataFrame({'key1':['foo','foo','bar','bar'],'key2':['one','one','one','two'],'ival':[4,5,6,7]})
* pd.merge(left,right,on='key1')
* pd.merge(left,right,on='key1',suffixes=('_left','_right'))

##### merge on index
* left1 = pd.DataFrame({'key':['a','b','a','a','b','c'],'value':range(6)})
* right1 = pd.DataFrame({'group_val':[3.5,7]}, index=['a','b'])
* pd.merge(left1,righ1,left_on='key',right_index=True)
* pd.merge(left1,right1,left_on='key',right_index=True, how='outer')
* left2=pd.DataFrame([[1,2],[3,4],[5,6]], index=['a','b','c'], columns=['Ohio','Nevada'])
* right2=pd.DataFrame([[7,8],[9,10],[11,12],[13,14]], index=['b','c','d','e'], columns=['Missouri','Albama'])
* pd.merge(left2,right2,how='outer',left_index=True,right_index=True)


#### join() method for merging by index
* left1 = pd.DataFrame({'key':['a','b','a','a','b','c'],'value':range(6)})
* right1 = pd.DataFrame({'group_val':[3.5,7]}, index=['a','b'])
* left1.join(righ1,on='key')  ==> perserving the left frame's row index
* left2=pd.DataFrame([[1,2],[3,4],[5,6]], index=['a','b','c'], columns=['Ohio','Nevada'])
* right2=pd.DataFrame([[7,8],[9,10],[11,12],[13,14]], index=['b','c','d','e'], columns=['Missouri','Albama'])
* another = pd.DataFrame([[7,8],[9,10],[11,12],[16,17]], index=['a','c','e','f'], columns=['New York','Oregon'])
* left2.join([right2, another])

### (3) concat() method

| Argument | Description          
| :- |-------------: 
| objs | List or dict of pandas objects to be concatenated; this is the only required argument
| axis | Axis to concatenate along; defaults to 0
| join | Either 'inner' or 'outer'; whether to intersection or union together indexes along the other axes
| join_axes | specific indexes to use for the other n-1 axes instead of performing union/intersection logic
| keys | Values to associated with objects being concatenated, forming a hierarchical index along the concatenation axis; can either be a list or array of arbitrary values, an array of tuples, or a list of arrays
| levels | Specific indexes to use as hierarchical index level or levels
| names | names for created hierarchical levels if keys and/or levels passed
| verify_integrity | check new axis in concatenated object for duplicates and raise exception if so; by default allow duplicates
| ignore_index | Do not preserve indexes along concatenation axis, instead producing a new range(total_length) index

* arr = np.arange(12).reshape(3,4)
* np.concatenate([arr,arr], axis=1)
* s1=pd.Series([0,1], index=['a','b'])
* s2=pd.Series([2,3,4], index=['c','d','e'])
* s3=pd.Series([5,6],index=['f','g'])
* pd.concat([s1,s2,s3])
* pd.concat([s1,s2,s3],axis=1)
* s4 = pd.concat([s1,s3])
* pd.concat([s1,s4])
* pd.concat([s1,s4],axis=1)
* pd.concat([s1,s4],axis=1, join='inner')
* pd.concat([s1,s4],axis=1, join_axes=[['a','c','b','e']])
* result=pd.concat([s1,s1,s3], keys=['one','two','three'])
* result.unstack()
* pd.concat([s1,s2,s3],axis=1,keys=['one','two','three']) ==> keys become the DataFrame column headers
* df1=pd.DataFrame(np.arange(6).reshape(3,2), index=['a','b','c'], columns=['one','two'])
* df2=pd.DataFrame(5+np.arange(4).reshape(2,2), index=['a','c'], columns=['three','four'])
* pd.concat([df1,df2],axis=1,keys=['level1','level2'])
* pd.concat({'level':df1, 'level2':df2}, axis=1)
* pd.concat([df1,df2], axis=1, keys=['level1','level2'], names=['upper','lower'])
* df1=pd.DataFrame(np.random.randn(3,4), columns=['a','b','c','d'])
* df2=pd.DataFrame(np.random.randn(2,3), columns=['b','d','a'])
* pd.concat([df1,df2],ignore_index=True)

### (4) combine_first() method
* a=pd.Series([np.nan,2.5,np.nan,3.5,4.5,np.nan],index=['f','e','d','c','b','a'])
* b=pd.Series(np.arange(len(a), dtype=np.float64),index=['f','e','d','c','b','a'])
* b[-1]=np.nan
* np.where(pd.isnull(a),b,a)
* b[:-2].combine_first(a[2:])  ==> patching missing data in the calling object with data from the object you pass

## 14. Reshaping and Pivoting
#### (1) Reshaping with Hierarchical Indexing

##### stack: this "rotate" or pivots from the columns in the data to the rows
##### unstack: this pivots from the rows into the columns
* data=pd.DataFrame(np.arange(6).reshape((2,3)), index=pd.Index(['Ohio','Colorado'],name='state'), columns=pd.Index(['one','two','three'], name='number'))
* result = data.stack()
* result.unstack()
* result.unstack(0)
* result.unstack('state')
* s1=pd.Series([0,1,2,3],index=['a','b','c','d'])
* s2=pd.Series([4,5,6], index=['c','d','e'])
* data2=pd.concat([s1,s2],keys=['one','two'])
* data2.unstack()
* data2.unstack().stack() ==> stacking filters out missing data by default
* data2.unstack().stack(dropna=False)
* df=pd.DataFrame({'left': result, 'right': result+5}, columns=pd.Index(['left','right'], name='side'))
* df.unstack('state')
* df.unstack('state').stack('side')

#### (2) Pivoting 'Long' to 'Wide' format
* pivoted = ldata.pivot('data','item','value')  ==>the first two argument can be used as the row and column index.
* ldata['value2'] = np.random.randn(len(ldata))
* pivoted = ldata.pivot('date','item')
* unstacked = ldata.set_index(['date','item']).unstak('item')

#### (3) Pivoting 'Wide' to 'Long' Format
pandas.melt: rather than transforming one column into many in a new DataFrame, it merges multiple columns into one
* df=pd.DataFrame({'key':['foo','bar','baz'],'A':[1,2,3],'B':[4,5,6],'C':[7,8,9]})
* melted = pd.melt(df,['key']) ==> 'key' column is a group indicator
* reshaped = melted.pivot('key','variable', 'value')
* pd.melt(df,id_vars=['key'],value_vars=['A','B'])
* pd.melt(df,value_vars=['A','B','C'])
* pd.melt(df,value_vars=['key','A','B'])

## 15. Group operations

### (1) GroupBy Mechanics
split-apply-combine. (1) Data contained in a panda object, is split into groups based on one or more keys. The splitting is performed on a particular axis of an object. (2) A function is applied to each group, producing a new value. (3)the results of all those functions applications are combined into a result object.

* df=pd.DataFrame({'key1':['a','a','b','b','a'],'key2':['one','two','one','two','one'], 'data1':np.random.randn(5),'data2':np.random.randn(5)})
* grouped=df['data1'].groupby(df['key1'])  ==> grouped is a GroupBy Object.
* grouped.mean()
* means=df['data1'].groupby([df['key1'],df['key2']]).mean()
* states = np.array(['Ohio','California','California','Ohio','Ohio'])
* years = np.array([2005,2005,2006,2005,2006])
* df['data1'].groupby([states,years]).mean()
* df.groupby('key1').mean()  ==>  no key2 column. Because df['key2'] is not a numeric data. By default, all of the numeric columns are aggregated.
* df.groupby(['key1','key2']).mean()
* df.groupby(['key1','key2']).size()

#### Iterating Over groups
* for name,group in df.groupby('key1'):
        print(name)
        print(group)
* for (k1,k2), group in df.groupby(['key1','key2']):
        print((k1,k2))
        print(group)
* pieces = dict(list(df.groupby('key1')))
* grouped=df.groupby(df.types, axis=1)
* for dtype, group in grouped:
        print(dtype)
        print(group)

#### Selecting a Column or Subset of Columns
* df.groupby('key1')['data1']  ==> df['data1'].groupby(df['key1'])
* df.groupby('key1')[['data2']]  ==> df[['data2']].groupby(df['key1'])
* df.groupby(['key1','key2'])[['data2']].mean()
* s_grouped = df.groupby(['key1','key2'])['data2']  ==> s_grouped is a grouped DataFrame

#### Grouping with Dicts and Series
* people=pd.DataFrame(np.random.randn(5,5), columns=['a','b','c','d','e'],index=['Joe','Steve','Wes','Jim','Travis'])
* people.iloc[2:3,[1,2]] = np.nan
* mapping={'a':'red','b':'red','c':'blue','d':'blue','e':'red', 'f':'orange'}
* by_column = people.groupby(mapping, axis=1)
* by_column.sum()
* map_series = pd.Series(mapping)
* people.groupby(map_series,axis=1).count()

#### Grouping with Functions
* people.groupby(len).sum()  ==> group by the length of the names.
* key_list=['one','one','one','two','two']
* people.groupby([len,key_list]).min()

#### Grouping by Index Levels
* columns = pd.MultiIndex.from_arrays([['US','US','US','JP','JP'],[1,3,5,1,3]],names=['cty','tensor'])
* hier_df=pd.DataFrame(np.random.randn(4,5), columns=columns)
* hier_df.groupby(level='cty',axis=1).count()

### (2) Data Aggregation

#### groupby method

| Argument | Description          
| :- |-------------: 
| count | Number of non-NA values in the group
| sum | Sum of non-NA values
| mean | Mean of non-NA values
| median | Arithmetic median of non-NA values
| std,var | Unbiased standard derivation and variance
| min,max | Minimum and maximum of non-NA values
| prod | Product of non-NA values
| first,last | First and last non-NA values

* df=pd.DataFrame({'key1':['a','a','b','b','a'],'key2':['one','two','one','two','one'], 'data1':np.random.randn(5),'data2':np.random.randn(5)})
* grouped=df.groupby('key1')
* grouped['data1'].quantile(0.9)
* def peak_to_peak(arr):
        return arr.max()-arr.min()
* grouped.agg(peak_to_peak)
* grouped.describe()

#### Column-Wise and Multiple Function Application
* tips = pd.read_csv('examples/tips.csv')
* tips['tip_pct'] = tips['tip']/tips['total_bill']
* grouped = tips.groupby(['day','smoker'])
* grouped_pct = grouped['tip_pct']
* grouped_pct.agg('mean')
* grouped_pct.agg(['mean','std','peak_to_peak'])
* grouped_pct.agg([('foo','mean'),('bar',np.std)])
* functions=['count','mean','max']
* result=grouped['tip_pct','total_bill'].agg(functions)
* result['tip_pct']
* grouped.agg({'tip':['min','max','mean','std'],'size':'sum'})

#### Return Aggregated Data Without Row Indexes
* tips.groupby(['day','smoker'],as_index=False).mean()
