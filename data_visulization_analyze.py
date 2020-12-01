import time
import datetime

import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pylab import rcParams

import patsy
import statsmodels.api as sm
import scipy.stats as stats
from scipy.stats import ttest_ind, chisquare, normaltest
from sklearn import metrics
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler

# Loading Dataset
df = pd.read_csv('googleplaystore.csv')


# Data clean, visualize and analyze
df.dropna(how='any', inplace=True)
df.drop_duplicates(['App'], inplace=True)

## Ratings Data Overview
df['Rating'].describe()
f, ax = plt.subplots(figsize=(10, 7))
sns.despine(f)
sns.kdeplot(df[df['Type']=='Free']['Rating'], color = 'b', shade=True)
sns.kdeplot(df[df['Type']=='Paid']['Rating'], color = 'r', shade=True)
plt.title('Rating Distribution', size=16)
plt.legend(['Free', 'Paid'], fontsize=12)
plt.xlabel('Ratings', size=12)
plt.ylabel('Frequency', size=12)

## Category - Rating
print(df['Category'].describe())
print('\nUnique Categories: \n', df['Category'].unique())

f, ax = plt.subplots(figsize=(15, 8))
sns.despine(f)
sns.countplot(x='Category', data = df)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
plt.title('Number of each Category', size=16)
plt.ylabel('Frequency', size=12)

f, ax = plt.subplots(figsize=(15, 8))
sns.despine(f)
sns.boxenplot(x='Category', y='Rating', data=df)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
plt.title('Rating vs Category', size=16)

## Reviews - Rating
df['Reviews'] = df['Reviews'].apply(lambda x : int(x))
df['Reviews'].describe()
f, ax = plt.subplots(figsize=(15, 8))
sns.despine(f)
sns.distplot(df[df['Reviews']<1e4]['Reviews'], kde=True)
plt.title('Reviews Distribution', size=16)
plt.ylabel('Frequency', size=12)

sns.jointplot('Reviews', 'Rating', df)

## Size - Rating
df.Size.unique()
df['Size'].replace('Varies with device', np.nan, inplace = True )
df.Size = (df.Size.replace(r'[kM]+$', '', regex=True).astype(float) * \
            df.Size.str.extract(r'[\d\.]+([KM]+)', expand=False)
            .fillna(1).replace(['k','M'], [10**3, 10**6]).astype(int))
df['Size'].fillna(df.groupby('Category')['Size'].transform('mean'),inplace = True)
sns.jointplot('Size', 'Rating', df)

## Install - Ratings
df_inst = df
installs_transform = {}
for i in list(df_inst['Installs'].unique()):
    installs_transform[i] = math.log(int(i[:-1].replace(',', '')))
df_inst = df_inst.replace({'Installs': installs_transform})
df_inst['Installs'].describe()
f, ax = plt.subplots(figsize=(15, 8))
sns.despine(f)
sns.regplot(x="Installs", y="Rating", data=df_inst);
plt.title('Rating VS Installs',size = 16)

## Type - Ratings
f, ax = plt.subplots(figsize=(15, 8))
sns.despine(f)
sns.boxenplot(x='Type', y='Rating', data=df)
plt.title('Type vs Ratings',size = 16)

plt.subplots(figsize=(15, 8))
plt.pie(df['Type'].value_counts(), explode=[0.1, 0.1], labels=['Free', 'Paid'], autopct='%1.1f%%', shadow=True)
plt.title('Percentage of each Type', size=16)

## Price - Ratings
df_paid = df[df['Type'] == 'Paid']
price_pair = {}
for price in list(df_paid['Price'].unique()):
    price_pair[price] = float(price[1:])
df_paid = df_paid.replace({'Price': price_pair})
f, ax = plt.subplots(figsize=(15, 8))
sns.despine(f)
sns.regplot(x="Price", y="Rating", data=df_paid[df_paid['Price']<100]);
plt.title('Price VS Ratings',size = 16)

## Genres - Ratings
df['Genres'].unique()
f, ax = plt.subplots(figsize=(15, 8))
sns.despine(f)
sns.countplot('Genres', data=df)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
plt.title('Genres Count', size=16)

df_genres = df
for genre in df['Genres'].unique():
    if df['Genres'].value_counts()[genre] < 100:
        df_genres = df_genres[df_genres['Genres']!=genre]
f, ax = plt.subplots(figsize=(15, 8))
sns.despine(f)
sns.countplot('Genres', data=df_genres)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
plt.title('Genres Count', size=16)

f, ax = plt.subplots(figsize=(15, 8))
sns.despine(f)
sns.boxenplot(x='Genres', y='Rating', data=df_genres)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
plt.title('Rating vs Genres', size=16)