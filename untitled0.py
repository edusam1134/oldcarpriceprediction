# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 17:17:19 2022

@author: Admin
"""

import pandas as pd

df= pd.read_csv("D:\project\carpricepridiction\Car-Price-Prediction-master\car data.csv")

df.head()
df.shape

print(df["Seller_Type"].unique())
print(df["Transmission"].unique())
print(df["Owner"].unique())

#checking null values
df.isnull().sum()

df.describe()
df.columns
final_dataset = df.iloc[:,1:]
final_dataset.columns
final_dataset.head()
final_dataset["Current_Year"]= 2022
final_dataset["no_year"] = final_dataset["Current_Year"] - final_dataset["Year"]
final_dataset.head()
final_dataset.drop(["Year","Current_Year"], axis=1 , inplace=True)
final_dataset.head()
final_dataset=pd.get_dummies(final_dataset, drop_first=True)
final_dataset.head()
final_dataset.corr()

import seaborn as sns
sns.pairplot(final_dataset)

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

corrmat=final_dataset.corr()
top_corr_features=corrmat.index
plt.figure(figsize=(20,20))
g=sns.heatmap(final_dataset[top_corr_features].corr(),annot=True,cmap='RdYlGn')

#input and output variable taking seprate
x=final_dataset.iloc[:,1:]
y=final_dataset.iloc[:,0]

x.head()
y.head()

####Feature Importance
from sklearn.ensemble import ExtraTreesRegressor
model=ExtraTreesRegressor()
model.fit(x,y)

print(model.feature_importances_)
# to plot graph for more top 5 important features #for use when more features present which to drop and which include 
feat_importances = pd.Series(model.feature_importances_, index=x.columns)
feat_importances.nlargest(5).plot(kind='barh')
plt.show()

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
x_train.shape
y_train.shape

import numpy as np
from sklearn.ensemble import RandomForestRegressor
rf_random = RandomForestRegressor()
n_estimators=[int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
print(n_estimators)

#Hypothisas
#randomized Search CV


# no of trees
n_estimators=[int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
print(n_estimators)
# no of feature to split
max_features = ['auto','sqrt']
#max num of level in tree
max_depth = [int(x) for x in np.linspace(5,30, num = 6)]
#min num of sample requiered to split a node
min_samples_split = [2,5,10,15,100]
#min num of sample required at each leaf node
min_samples_leaf = [1,2,5,10]

from sklearn.model_selection import RandomizedSearchCV
# create the  random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf
              }

print(random_grid)

#creating base model
rf = RandomForestRegressor()

rf_random = RandomizedSearchCV(estimator = rf, param_distributions= random_grid, scoring = 'neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)

rf_random.fit(x_train,y_train)

predictions= rf_random.predict(x_test)

predictions

sns.distplot(y_test-predictions)

plt.scatter(y_test,predictions)

import pickle
file = open('random_forest_regression_model.pkl','wb')

#dump info to file
pickle.dump(rf_random, file)

import os

os.getcwd()
