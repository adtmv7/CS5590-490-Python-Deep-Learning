# Importing libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error, r2_score

# Reading input data pertaining to Boston Housing
boston = pd.read_csv('trainBoston.csv')
boston.medv.describe()

# Identifying features and predictor and split the data set into training and test set
x = boston.drop(['medv'], axis=1)
y = boston[['medv']]

# Building regression model
lr = linear_model.LinearRegression()
model = lr.fit(x, y)
print(model)

#Evaluating before applying Exploratory Data Analysis
prediction = model.predict(x)
print("R^2 before applying EDA: %.2f" % r2_score(y,prediction))
print("RMSE before applying EDA: %.2f" % mean_squared_error(y,prediction))

# Handling null values within the data set
nulls = pd.DataFrame(boston.isnull().sum().sort_values(ascending=False)[:25])
nulls.columns = ['Null Count']
nulls.index.name = 'Feature'
print(nulls)

# Here we are filling the null values with mean value
train_data=boston.apply(lambda x: x.fillna(x.mean()),axis=0)
print(train_data["medv"])
print(train_data.isnull().sum())

#split the data set into training and test set

x_train = train_data.drop(['medv'], axis=1)
y_train = train_data['medv']
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=.5)

# Building regression model
lr = linear_model.LinearRegression()
model = lr.fit(x_train, y_train)
print(model)

#Evaluating after applying Exploratory Data Analysis
prediction = model.predict(x_train)
print("R^2 after applying EDA: %.2f" % r2_score(y_train,prediction))
print("RMSE after applying EDA: %.2f" % mean_squared_error(y_train,prediction))


