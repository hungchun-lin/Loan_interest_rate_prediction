#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Load required package.
# Packages for data preprocessing and data visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import datetime as dt
import dataframe_image as dfi
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Ignore warnings to improve readibility.
import warnings
warnings.filterwarnings("ignore")


# In[2]:


# Load Data
df = pd.read_csv("loans_full_schema.csv")


# In[3]:


# Look at the head of the data
df.head()


# In[71]:


# Check the data type and data status in each columns
data_type = pd.concat([df.nunique(), df.dtypes, df.isnull().sum(), df.describe(include = "all")           .T[['min', 'max']]], axis = 1).reset_index()            .rename(columns = {"index": "field",
                               0: "num_of_unique_value",
                               1: "data_type",
                               2: "count_of_null",
                               3: "min",
                               4: "max"})
data_type


# In[6]:


# Data Structure
print(f'Number of records: {df.shape[0]}')
print(f'Number of fields in each record: {df.shape[1]}')


# In[7]:


# The function for checking missing value in each column
def missing_value_checker(df):
    variable_proportion = [
            [variable, round((df[variable].isna().sum() / df.shape[0])*100, 4)] 
            for variable in df.columns 
            if df[variable].isna().sum() >= 0
        ]

    print('%-30s' % 'Variable with missing values', 'Percentage of missing values')
    for variable, proportion in sorted(variable_proportion, key=lambda x : x[1]):
        print('%-30s' % variable, proportion)
    
    variable_proportion = pd.DataFrame(variable_proportion, columns = ['Variable', 'missing%'])
        
    return variable_proportion


# In[8]:


variable_proportion = missing_value_checker(df)


# In[9]:


# Plot the homeowenership vs avg interest rate

fig, ax1 = plt.subplots(figsize= (8, 4))
ax2 = ax1.twinx()

sns.countplot(data=df, x='homeownership', ax=ax1)
pvt = pd.pivot_table(df, values='interest_rate', index='homeownership', aggfunc='mean').reset_index()
sns.lineplot(data=pvt, x='homeownership', y='interest_rate', ax=ax2, color='#555555', lineWidth=5, markers=10)

ax1.grid(False)
ax2.grid(False)
ax1.set_xlabel('homeownership', size=12)
ax1.set_ylabel('homeownership count', size=12)
ax2.set_ylabel('average interest rate', size=12)

fig.show()


# In[10]:


# Plot the count of the employee title

plt.figure(figsize=(10, 4))

loan_purpose_cnt = pd.pivot_table(df, values='emp_title', index='loan_purpose', aggfunc='count').reset_index().    rename(columns={'emp_title': 'count'}).sort_values(by='count', ascending=False)
sns.barplot(data=loan_purpose_cnt, x='loan_purpose', y='count')
plt.xticks(rotation=45, ha='right')
plt.show()


# In[11]:


# plot the box plot to see the loan amount in each loan status

ax = plt.subplots(figsize = (10, 6))
ax = sns.boxplot(x='loan_status', y='loan_amount', data=df)
sns.despine()


# In[12]:


# Plot the box plot of the interest rate for each grade

ax = plt.subplots(figsize = (10, 6))
ax = sns.boxplot(x='grade', y='interest_rate', data=df, order = sorted(df['grade'].unique()))
sns.despine()


# In[16]:


# plot the histogram for the loan amount

plt.figure(figsize=(10, 6))
n, bins, patches = plt.hist(df['loan_amount'], align='mid', edgecolor='white', color='lightblue')
for num in bins:
    plt.text(num, 1, round(num, 1), ha='center', color='black')
plt.xlabel('Loan Amount')
plt.show()


# In[15]:


# plot the histogram for the interest rate

plt.figure(figsize=(10, 6))
n, bins, patches = plt.hist(df['interest_rate'], align='mid', edgecolor='white', color='lightblue')
for num in bins:
    plt.text(num, 1, round(num, 1), ha='center', color='black')
plt.xlabel('Loan Amount')
plt.show()


# In[45]:


# Check the missing value and consider to drop and deal with the missing value
variable_proportion[variable_proportion['missing%'] != 0]


# In[46]:


drop_col = list(variable_proportion[variable_proportion['missing%'] > 50]['Variable'])


# In[47]:


drop_col.append('emp_title')
drop_col.append('emp_length')


# In[48]:


df = df.drop(columns = drop_col)


# In[49]:


variable_proportion[variable_proportion['missing%'] != 0]


# In[50]:


df['months_since_last_credit_inquiry'].fillna(int(df['months_since_last_credit_inquiry'].mean()), inplace=True)
df['num_accounts_120d_past_due'].fillna(int(df['num_accounts_120d_past_due'].mean()), inplace=True)
df = df.dropna() # for debt_to_income


# In[51]:


df.isnull().sum()


# In[52]:


df_float_col = pd.DataFrame(df.drop(columns = ['interest_rate']).dtypes[df.drop(columns = ['interest_rate']).dtypes == 'float64']).index
df_float = df[df_float_col]
  
vif_data = pd.DataFrame()
vif_data["feature"] = df_float.columns
  
# calculating VIF for each feature
vif_data["VIF"] = [variance_inflation_factor(df_float.values, i)
                          for i in range(len(df_float.columns))]



# In[53]:


vif_data


# In[54]:


vif_data[vif_data['VIF'] > 5]


# In[55]:


sns.set(style="whitegrid", font_scale=1)

plt.figure(figsize=(20,20))
plt.title('Pearson Correlation Matrix',fontsize=25)
sns.heatmap(df.corr(),linewidths=0.25,vmax=0.7,square=True,cmap="GnBu",linecolor='w',
            annot=True, annot_kws={"size":10}, cbar_kws={"shrink": .7})


# In[56]:


# Start training process
import math
import xgboost
from xgboost import XGBRegressor
from itertools import chain, combinations
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, LassoCV
from sklearn.model_selection import cross_val_score, train_test_split

from sklearn.preprocessing import MinMaxScaler


# In[57]:


# Encoding the categorical data columns

obj_columns = [col for col in df.columns if df[col].dtype.name == 'object']
df = pd.get_dummies(df, columns = obj_columns)


# In[58]:


X = df.drop('interest_rate', axis=1)
y = df['interest_rate']


# In[60]:


print(X.shape)
print(y.shape)


# In[61]:


# split the data into train and test 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[62]:


# Train the linear regression model
lm = LinearRegression()
lm.fit(X_train,y_train)


# In[63]:


lm_y_pred = lm.predict(X_test)


# In[64]:


mse = mean_squared_error(y_test, lm_y_pred)
RMSE = math.sqrt(mse)
mae = mean_absolute_error(y_test, lm_y_pred)
r2 = r2_score(y_test, lm_y_pred)

print(f'MSE: {mse}')
print(f'RMSE: {RMSE}')
print(f'MAE: {mae}')
print(f'R^2: {r2}')


# In[67]:


# Check for Linearity
f = plt.figure(figsize=(14,5))
ax = f.add_subplot(121)
sns.scatterplot(y_test,lm_y_pred,ax=ax,color='r')
ax.set_title('Check for Linearity:\n Actual Vs Predicted value')

# Check for Residual normality & mean
ax = f.add_subplot(122)
sns.distplot((y_test - lm_y_pred),ax=ax,color='b')
ax.axvline((y_test - lm_y_pred).mean(),color='k',linestyle='--')
ax.set_title('Check for Residual normality & mean: \n Residual eror')


# In[68]:


# Train the XGB Regressor model
xgb = XGBRegressor(n_estimators=1000, max_depth=7, eta=0.1, subsample=0.7, colsample_bytree=0.8)
xgb.fit(X_train,y_train)


# In[69]:


xgb_y_pred = xgb.predict(X_test)


# In[70]:


mse = mean_squared_error(y_test, xgb_y_pred)
RMSE = math.sqrt(mse)
mae = mean_absolute_error(y_test, xgb_y_pred)
r2 = r2_score(y_test, xgb_y_pred)

print(f'MSE: {mse}')
print(f'RMSE: {RMSE}')
print(f'MAE: {mae}')
print(f'R^2: {r2}')

