# Importing libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense

# Reading data
breastCancer = pd.read_csv('BreastCancer.csv')

# Converting non-numerical data into numerical
breastCancer["diagnosis"] = pd.Categorical(breastCancer["diagnosis"])
breastCancer["diagnosis"] = breastCancer["diagnosis"].cat.codes
cancerData = breastCancer.values

# Split the data set into training and test sets
x_train, x_test, y_train, y_test = train_test_split(cancerData[:, 2:32], cancerData[:, 1], test_size=0.2, random_state=45)

# Data is normalize here
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

nnCancer = Sequential() #Creating model
nnCancer.add(Dense(20, input_dim=30, activation='relu')) # first hidden input dense layer
nnCancer.add(Dense(1, activation='sigmoid')) #Define the output neuron

# Fitting the neural network model on the training data set
nnCancer.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
nnCancerModel = nnCancer.fit(x_train, y_train, epochs=100, verbose=0, initial_epoch=0)

# Display the neural network identified
print('The Summary of the Neural Model is', nnCancer.summary())
print(nnCancer.evaluate(x_test, y_test, verbose=0))


