import re
import sys

import time
import datetime

import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Loading the data
df = pd.read_csv("GooglePlay_pion.csv")
df.to_csv('original.csv')
print(df.info())

# Insert missing rating value
df['Rating'] = df['Rating'].fillna(df['Rating'].median())

# Count the number of unique values in category column
df['Category'].unique()
# Removing NaN values
df = df[pd.notnull(df['Last Updated'])]
df = df[pd.notnull(df['Content Rating'])]
print(df.isnull().sum())

# Encode app features
le = preprocessing.LabelEncoder()
df['App'] = le.fit_transform(df['App'])

# Encode category features
category_list = df['Category'].unique().tolist()
category_list = ['cat_' + word for word in category_list]
df = pd.concat([df, pd.get_dummies(df['Category'], prefix='cat')], axis=1)
print(len(category_list))
# Encode genres features
le = preprocessing.LabelEncoder()
df['Genres'] = le.fit_transform(df['Genres'])

# Encode content rating features
le = preprocessing.LabelEncoder()
df['Content Rating'] = le.fit_transform(df['Content Rating'])

# Price Encoding
df['Price'] = df['Price'].apply(lambda x: x.strip('$'))

# Type encoding
df['Type'] = pd.get_dummies(df['Type'])
# Last Updated encoding
df['Last Updated'] = df['Last Updated'].apply(lambda x : time.mktime(datetime.datetime.strptime(x, '%B %d, %Y').timetuple()))

# Installs Encoding
df['Installs'] = df['Installs'].apply(lambda x : x.strip('+').replace(',', ''))
df['Installs'] = df['Installs'].astype(float)
# 1:4 point to the next level
df['Installs'] = df['Installs'].apply(lambda x : 1.8*float(x))

# Preprocessing Size columns
# Change kbytes to Mbytes
k_indices = df['Size'].loc[df['Size'].str.contains('k')].index.tolist()
converter = pd.DataFrame(df.loc[k_indices, 'Size'].apply(lambda x: x.strip('k')).astype(float).apply(lambda x: x / 1024).apply(lambda x: round(x, 3)).astype(str))
df.loc[k_indices,'Size'] = converter
# Remove Size with Varies with device
df['Size'] = df['Size'].apply(lambda x: x.strip('M'))
i = df[df['Size'] == 'Varies with device'].index
df=df.drop(i)
df['Size'] = df['Size'].astype(float)
df.to_csv('modified.csv')
# Split data into training and testing sets
features = ['App', 'Rating', 'Reviews', 'Size', 'Type', 'Price', 'Content Rating', 'Genres', 'Last Updated']
features.extend(category_list)
X = df[features]
print(len(features))
y = df['Installs']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 10)

# Scaling trained data
scaler = StandardScaler().fit(X_train)
rescaled_X_train = scaler.transform(X_train)

# Choose the best regressor
#model = AdaBoostRegressor(random_state=0, n_estimators=1000)
model = GradientBoostingRegressor(random_state=21, n_estimators=400)
#model= LinearRegression()

model.fit(rescaled_X_train, y_train)
accuracy1 = model.score(rescaled_X_train, y_train)

rescaled_X_test = scaler.transform(X_test)
diabetes_y_pred = model.predict(rescaled_X_test)
print(diabetes_y_pred)
def rmsle(y, y0):
    return np.sqrt(np.mean(np.square(np.log1p(y) - np.log1p(y0))))
result = rmsle(y_test, diabetes_y_pred)
print(result)
print('Error Metrics: ' + str(result))
accuracy = model.score(rescaled_X_test, y_test)

print('Train Accuracy: ' + str(np.round(accuracy1*100, 2)) + '%')
print('Validation Accuracy: ' + str(np.round(accuracy*100, 2)) + '%')
