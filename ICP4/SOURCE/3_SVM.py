# Importing libraries as needed
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import SVC

# Reading the provided data set
# Defining the training and test
train_df = pd.read_csv('glass.csv')
X = train_df.drop("Type", axis=1)
Y = train_df["Type"]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

# Fitting Naive bayes model
svc = SVC()
svc.fit(X_train, y_train)
Y_prediction= svc.predict(X_test)
acc_svc = round(svc.score(X_train, y_train) * 100,2)

# Computing the error rate of the model fit
print("SVM accuracy is:", acc_svc)
print(classification_report(y_test, Y_prediction))
