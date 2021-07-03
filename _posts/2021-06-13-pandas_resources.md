---
toc: true
layout: post
description: Pandas code snippets and recipes that I revisit now and again.
categories: [exploratory-data-analysis, machine-learning, resources]
title: Pandas cookbook
---

Selected Pandas code snippets and recipes that I revisit now and again. The snippets are adopted from different python scripts written over time, ignore the variable names.

This post was inspired by the wonderful work of [Chris Albon](https://chrisalbon.com/#python) and his snippets of code blocks. 

> Last update: 13th June 2021

## Reading and Writing 

**Basic reading a blank csv file**
```python
# If you have no column names but know the number of columns
pd.read_csv(file_name, header=None, names=['col1','col2'])
```

**Saving a file to not have 'Unamed' column**
```python
df1.to_csv(os.path.join(output_dir, 'file_name_to_save_as.csv'), sep=',',columns=df1.columns, index=False, header=False) # header = None for no column names
```


**Information about the dataframe**
```python
pandas_dataframe.info()
```

**Summary statistics (mean, quartile ranges)**
```python
pandas_dataframe.describe().round(2)
```

**Replace**
```python
df = df.replace( [list_of_value_to_replace], value_to_replace_with)
# Eg: df.replace( [98-99], np.nan)
```

**Replace characters in the columns**

```python
# List of characters to remove
chars_to_remove = ['+','$',',']

# List of column names to clean
cols_to_clean = ['Installs','Price']

# Loop for each column in cols_to_clean
for col in cols_to_clean:
    # Loop for each char in chars_to_remove
    for char in chars_to_remove:
        # Replace the character with an empty string
        apps[col] = apps[col].apply(lambda x: x.replace(char, ''))
    # Convert col to float data type
    apps[col] = apps[col].astype(float)
```

**Convert spaces titles in the row to one word separated by '-'**
```python
reduced_df['product_title'] = reduced_df['product_title'].apply( lambda x: x.lower().replace(' ', '-') )
```

**Define a new column with temp entries** 
```python
pandas_dataframe['columns_name'] = 42 
```
 
**Create columns in a loop**
```python
pandas_dataframe.columns = ['feature_' + str(i) for i in range(n_columns)]
```

**Dropping miscellaneous columns and NaN entries**
```python
columns_to_drop = ['CookTimeInMins', 'Servings', 'Course', 'Diet', 'Instructions', 'TranslatedInstructions', 'URL']
food_df = food_df.drop(columns = columns_to_drop).dropna()
```

## Quick plotting 

**Simple pearson correlation plot**

```python
# Generate Pearson Correlation Matrix for HOUSING 
corr_matrix=housing.corr()

# Edit the visuals and precision 
corr_matrix.style.background_gradient(cmap='coolwarm').set_precision(2)

# Look at Pearson values for one attribute 
corr_matrix['median_house_value'].sort_values(ascending=True)
```

**Plot multiple scatter plots**
```python
from pandas.plotting import scatter_matrix
attributes = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]
scatter_axes = scatter_matrix(housing[attributes], figsize=(12, 8));
```

## Handling missing values

**Option A: Dropping values in the columns with NaN**
```python
housing.dropna(subset=["total_bedrooms"])
```

**Option B: Drop that column entirely**
```python
housing.drop("total_bedrooms", axis=1)
```

**Option C: Fill missing value with some central tendency** 
```python
attribute_median = housing["total_bedrooms"].median()
housing["total_bedrooms"].fillna( attribute_median, inplace=True ) 
```

**Checking the NULL enties in the dataset**
```python
sample_incomplete_rows = housing[housing.isnull().any(axis=1)].head()
```

**Get number of NULL entries in the dataframe columns**
```python
null_columns=food_df.columns[food_df.isnull().any()]
food_df[null_columns].isnull().sum()
```


**Print full rows having NULL entries in the df**
```python
is_NaN = food_df.isnull()
row_has_NaN = is_NaN.any(axis=1)
rows_with_NaN = food_df[row_has_NaN]
```

**Dropping NULL only from a particular column**
```python
df_income_drop_na = df.dropna(subset=['INCOME2'])
```


## Join two datasets

**1. Inner join**

Only returns rows with matching values in both df. 

```python
A_B = A.merge(B, on = <common column name>, suffixes = tuples to append the name of columns with similar names) 
```

Remember that .merge() only returns rows where the values match in both tables.

**2. Merging more than one table**

```python
df1.merge(df2, on='col_A') \
    .merge(df3, on='col_B') \
    .merge(df4, on='col_C')
```

**3. Merge across multiple columns tags**
```python
df1.merge( df2, on = ['col_A', 'col_B'])
```
  
## Searching 

**Find columns names based on a string**
```python
df_raw_data.columns[df_raw_data.columns.str.contains('STRING_SUBSET')]
```

**Filter entries in the column based on the threshold** 
* Data has indian-inspired international cuisines which are not what we are interested in
```python
cuisin_counts = food_df['Cuisine'].value_counts()
cuisin_counts_more_than_50 = cuisin_counts.iloc[np.where(cuisin_counts > 50)]
food_df_top_cuisine = food_df.loc[ food_df['Cuisine'].isin(list(cuisin_counts_more_than_50.index))  ] 
#Dropping entries in `food_df` which have non-ind
```

**Clean up entries with partial matches**
```python
df.loc[df['Store Name'].str.contains('Wal', case=False), 'Store_Group_1'] = 'Walmarts'
```

```python
south_indian_tag = ['Chettinad', 'Andhra', 'Karnataka', 'Tamil Nadu', 'Kerala Recipes', 'South Indian Recipes']
food_df_top_cuisine.loc[food_df_top_cuisine['Cuisine'].isin(south_indian_tag), 'Combined_cuisine'] = 'South Indian'
```

**With `or` statements**

```python
String_filter_option = ['cond_1', 'cond_2']
pandas_dataframe[ Pandas_dataframe[ 'columns' ].str.contains('|'.join(string_filter_option)) ] 
```

**Filter rows in the pandas df with another list**
```python
month_list = ['May','Jun','Jul','Aug','Sep']
df_pH.loc[df_pH['Month'].isin(month_list)]
```

**Filter out  values using names: Making a separate list of those that DO NOT satisfy the constraint**
```python
no_bands = halftime_musicians[ ~halftime_musicians.musician.str.contains('Marching') ]
```

## Statistics & Distributions

**Histogram**
```python
df.hist('WTKG3')
```

**CDF and PDF** 

```python
# Functions for PMF and CDF, we will come to those later in the notebook 
def pmf(pandas_series):
    values, counts = np.unique(pandas_series, return_counts = True)
    pmf = np.c_[ values, counts / sum(counts) ]

    return pmf 

def cdf(pandas_series):
    values, counts = np.unique(pandas_series, return_counts = True)
    pmf = np.c_[ values, counts / sum(counts) ]
    cdf = np.zeros(shape=pmf.shape) 
    
    for i in range(0, pmf.shape[0]):
        cdf[i] = [pmf[i][0], np.sum(pmf[:i+1], axis=0)[-1]] 
        
    return cdf
```

**Confidence interval**

A bootstrap analysis of the reduction of deaths due to handwashing

```python
boot_mean_diff = []
for i in range(3000):
    boot_before = before_proportion.sample(frac=1, replace=True)
    boot_after = after_proportion.sample(frac=1, replace=True)
    boot_mean_diff.append( boot_after.mean() - boot_before.mean() )
```

Calculating a 95% confidence interval from `boot_mean_diff`

```python
confidence_interval = pd.Series(boot_mean_diff).quantile([0.025,0.975])
```

## Convert variables 

**Convert continuous variable to discrete**

```python
pd.cut 
```

Example 1:
```python
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf], #bins around 2-5 income bracket
                               labels=[1, 2, 3, 4, 5])
```

Use cut when you need to segment and sort data values into bins. This function is also useful for going from a continuous variable to a categorical variable. For example, cut could convert ages to groups of age ranges. Supports binning into an equal number of bins, or a pre-specified array of bins.

Example 2: 
```python
pd.cut(iris_df['sepal_length'], bins=3, right=True, labels=['low','med','high'], retbins=True)
```

**Fine tune the labeling**

```python
def convert_to_cat(panda_series):
    first_quarter = panda_series.describe()['25%']
    third_quarter = panda_series.describe()['75%']
    print(first_quarter, third_quarter)
    
    cat_list = ['temp'] * len(panda_series) 

    for i, entry in enumerate(panda_series):
        if entry <= first_quarter: 
            cat_list[i] = 'SMALL'
        elif first_quarter < entry <= third_quarter:
            cat_list[i] = 'MED'
        else:
            cat_list[i] = 'LARGE'
    
    return cat_list
```
                                                   
**Cateogorical variables to one-hot**
```python
# Pandas get dummies is one option 
pd.get_dummies(iris_df['sepal_width_cat'], prefix='sepal_width')
```

**One-hot discrete variable with more granularity**
```python
def OHE_discreet(point, pandas_series, intervals):
    '''
    define range for one-hot, for every entry find the closest value in the one-hot
    '''

    z = np.linspace(min(pandas_series), max(pandas_series), intervals)
    ohe = np.zeros(len(z))
    ohe[np.argmin(abs(z - point)**2)] = 1
    return ohe

iris_df['sepal_width_OHE'] = iris_df['sepal_width'].apply(OHE_discreet, args=(iris_df['sepal_width'], 11))
```

## Grouping data by entries in a row:

**Example 1**
```python
licenses_zip_ward.groupby('alderman').agg({'income':'median'})
```

Estimate the statistic of 'income' after grouping the dataframe by row entries in column 'alderman'

**Example 2**
```python
counted_df = licenses_owners.groupby('title').agg({'account':'count'})
```

I want to know the number of account each unique title entry has in the df. Here the column `account` was counted and the total entries were reported when the data frame was first grouped by entries in the title column. 

**Example 3**

Groupby multiple columns and show the counts

```python
# Create a column that will store the month
data['month'] = data['date'].dt.month

# Create a column that will store the year
data['year'] = data['date'].dt.year

# Group by the month and year and count the pull requests
counts = data.groupby(['month','year'])['pid'].count()
```

**Example 4**

Group and pivot table. Find the number of pull_request for the repo every year for the two authors: 

```python
# The developers we are interested in
authors = ['xeno-by', 'soc']

# Get all the developers' pull requests
by_author = pulls.loc[ pulls['user'].isin(authors) ]
by_author['year'] = by_author['date'].dt.year 

# Count the number of pull requests submitted each year
counts = by_author.groupby(['user', 'year']).agg({'pid': 'count'}).reset_index()

# Convert the table to a wide format
counts_wide = counts.pivot_table(index='year', columns='user', values='pid', fill_value=0)

# Plot the results
counts_wide
```






