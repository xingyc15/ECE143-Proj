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

#preprocessing
df = pd.read_csv('googleplaystore.csv')
df = df.drop(['Current Ver', 'Genres', 'Android Ver'], axis = 1)
df = df.dropna()
df = df[(df['Content Rating'] == 'Everyone')].copy()

def standardize_Installs(inputstring):
    outputstring = inputstring.replace('+', '')
    outputstring = outputstring.replace(',', '')
    outputstring = int(outputstring)
    return outputstring
def standardize_Reviews(inputstring):
    outputstring = int(inputstring)
    return outputstring
def standardize_Price(inputstring): 
    outputstring = inputstring.replace('$', '') 
    outputstring = float(outputstring)
    return outputstring
def standardize_Type(inputstring):
    if inputstring == 'Free':
        outputstring = 0
    else:
        outputstring = 1
    return outputstring

df['Installs'] = df['Installs'].apply(standardize_Installs)
df['Reviews'] = df['Reviews'].apply(standardize_Reviews)
df['Price'] = df['Price'].apply(standardize_Price)
df['Type'] = df['Type'].apply(standardize_Type)
df = pd.get_dummies(df, columns=['Category'])

df['Last Updated'] = df['Last Updated'].apply(lambda x : time.mktime(datetime.datetime.strptime(x, '%B %d, %Y').timetuple()))
k_indices = df['Size'].loc[df['Size'].str.contains('k')].index.tolist()
converter = pd.DataFrame(df.loc[k_indices, 'Size'].apply(lambda x: x.strip('k')).astype(float).apply(lambda x: x / 1024).apply(lambda x: round(x, 3)).astype(str))
df.loc[k_indices,'Size'] = converter
df['Size'] = df['Size'].apply(lambda x: x.strip('M'))
df = df[(df['Size'] != 'Varies with device')].copy()
df['Size'] = df['Size'].astype(float)

# select features that may affect the rate
X = df.drop(labels = ['App', 'Rating', 'Content Rating'], axis = 1)
y = df.Rating

# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 10)
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# KNN Model
model1 = KNeighborsRegressor(n_neighbors=15)
model1.fit(X_train, y_train)
y_predict1 = model1.predict(X_test)

# visualize predictions
plt.figure(figsize = (12, 7))
x = np.arange(0, len(y_test), 10)
p1 = plt.plot(x, np.array(y_test)[x], marker = 'x', label = 'Actual Ratings')
p2 = plt.plot(x, np.array(y_predict1)[x], marker = '.', label = 'Predicted Ratings')
plt.legend(loc = 1)
plt.grid()
plt.title('KNN Model')
plt.xlabel(r'$\mathit{i}$' + ' th sample')
plt.ylabel('Ratings')
plt.show()

# evaluation 
ACC1 = model1.score(X_test, y_test)
MSE1 = metrics.mean_squared_error(y_test, y_predict1)
print('R^2 = ' + str(ACC1))
print('MSE = ' + str(MSE1))

# Linear Regression Model
model2 = LinearRegression()
model2.fit(X_train, y_train)
y_predict2 = model2.predict(X_test)

# visualize predictions
plt.figure(figsize = (12, 7))
x = np.arange(0, len(y_test), 10)
p1 = plt.plot(x, np.array(y_test)[x], marker = 'x', label = 'Actual Ratings')
p2 = plt.plot(x, np.array(y_predict2)[x], marker = '.', label = 'Predicted Ratings')
plt.legend(loc = 1)
plt.grid()
plt.title('Linear Model')
plt.xlabel(r'$\mathit{i}$' + ' th sample')
plt.ylabel('Ratings')
plt.show()

# evaluation
ACC2 = model2.score(X_test, y_test)
MSE2 = metrics.mean_squared_error(y_test, y_predict2)
print('R^2 = ' + str(ACC2))
print('MSE = ' + str(MSE2))

# SVM Model
model3 = svm.SVR()
model3.fit(X_train, y_train)
y_predict3 = model3.predict(X_test)

# visualize predictions
plt.figure(figsize = (12, 7))
x = np.arange(0, len(y_test), 10)
p1 = plt.plot(x, np.array(y_test)[x], marker = 'x', label = 'Actual Ratings')
p2 = plt.plot(x, np.array(y_predict3)[x], marker = '.', label = 'Predicted Ratings')
plt.legend(loc = 1)
plt.grid()
plt.title('SVM Model')
plt.xlabel(r'$\mathit{i}$' + ' th sample')
plt.ylabel('Ratings')
plt.show()

# evaluation
ACC3 = model3.score(X_test, y_test)
MSE3 = metrics.mean_squared_error(y_test, y_predict3)
print('R^2 = ' + str(ACC3))
print('MSE = ' + str(MSE3))

# Random Forest Model
model4 = RandomForestRegressor(n_estimators = 500)
model4.fit(X_train, y_train)
y_predict4 = model4.predict(X_test)

# visualize predictions
plt.figure(figsize = (12, 7))
x = np.arange(0, len(y_test), 10)
p1 = plt.plot(x, np.array(y_test)[x], marker = 'x', label = 'Actual Ratings')
p2 = plt.plot(x, np.array(y_predict4)[x], marker = '.', label = 'Predicted Ratings')
plt.legend(loc = 1)
plt.grid()
plt.title('Random Forest Model')
plt.xlabel(r'$\mathit{i}$' + ' th sample')
plt.ylabel('Ratings')
plt.show()

# evaluation
ACC4 = model4.score(X_test, y_test)
MSE4 = metrics.mean_squared_error(y_test, y_predict4)
print('R^2 = ' + str(ACC4))
print('MSE = ' + str(MSE4))

# plot feature weights
feature_weights = pd.Series(data = model4.feature_importances_, index = X.columns)
plt.figure(figsize = (14, 10))
feature_weights.sort_values().plot.barh()
plt.show()

# compare models with metrics R^2 and MSE
model_names = ['KNN Model', 'Linear Model', 'SVM Model', 'Random Forest Model']
ACC = pd.Series(data = [ACC1, ACC2, ACC3, ACC4], index = model_names)
MSE = pd.Series(data = [MSE1, MSE2, MSE3, MSE4], index = model_names)
plt.figure(figsize=(12,7))
plt.subplot(2,1,1)
ACC.sort_values().plot.barh()
plt.title('Coefficient of Determinant')
plt.subplot(2,1,2)
MSE.sort_values().plot.barh()
plt.title('Mean Squared Error')
plt.show()