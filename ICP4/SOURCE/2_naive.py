# Importing libraries as needed
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# Reading the provided data set
train_df = pd.read_csv('glass.csv')
X = train_df.drop("Type", axis=1)
Y = train_df["Type"]

# Defining the training and test data sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

# Fitting Naive bayes model
model = GaussianNB()

# Predicting the results of the model on the test data
Y_prediction = model.fit(X_train, y_train).predict(X_test)
acc_model = round(model.score(X_test, y_test) * 100)

# Computing the error rate of the model fit
print("Model accuracy is:", acc_model)
print(classification_report(y_test, Y_prediction))