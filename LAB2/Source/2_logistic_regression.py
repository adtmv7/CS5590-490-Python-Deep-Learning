# Importing libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.optimizers import Adam, SGD
import numpy as np
from keras.layers import Dense
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from keras.callbacks import TensorBoard

# Reading data
heartData = pd.read_csv('heart.csv')

# Identifying features and predictor associated with the heart data set
x = heartData.iloc[:, 0:13]
y = heartData.iloc[:, 13]

# Split the data set into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=.33)

# Normalization of training data set
sc = StandardScaler()
scData = sc.fit(x_train, x_test)
print(scData)

# Hyper parameters set 1
# activation_function='relu'
# learning_rate=0.1
# epochs=50
# b_size=256
# decay_rate= learning_rate / epochs
# optimizer = Adam(lr=learning_rate, decay=decay_rate)

# Hyper parameters set 2
# activation_function='tanh'
# learning_rate=0.3
# epochs=100
# b_size=32
# decay_rate= learning_rate / epochs
# optimizer = SGD(lr=learning_rate, decay=decay_rate)

# Hyper parameters set 3
activation_function='relu'
learning_rate=0.5
epochs=150
b_size=60
decay_rate= learning_rate / epochs
optimizer = SGD(lr=learning_rate, decay=decay_rate)

# Creating neural network model for evaluation
# Define the model used
nnHeart = Sequential()
# Provide input and neurons for first hidden dense layer
nnHeart.add(Dense(60, input_dim=13, activation=activation_function))
# Adding multiple hidden layers
nnHeart.add(Dense(30, activation='relu'))
nnHeart.add(Dense(15, activation='tanh'))
# Define the output neuron
nnHeart.add(Dense(1, activation='sigmoid'))
# Fitting the neural network model on the training data set
nnHeart.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Fitting the model defined using the training data along with validation using test data
history = nnHeart.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, verbose=0, initial_epoch=0)

# Evaluation of the loss and accuracy associated to the test data set
[test_loss, test_acc] = nnHeart.evaluate(x_test, y_test)
print("Evaluation result on Test Data : Loss = {}, accuracy = {}".format(test_loss, test_acc))

# Listing all the components of data present in history
print('The components of data present in history are', history.history.keys())

# Graphical evaluation of accuracy associated with training and validation data
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Evaluation of Data Accuracy')
plt.xlabel('epoch')
plt.ylabel('Accuracy of Data')
plt.legend(['TrainData', 'ValidationData'], loc='upper right')
plt.show()

# Graphical evaluation of loss associated with training and validation data
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('Loss of Data')
plt.title('Evaluation of Data Loss')
plt.legend(['TrainData', 'ValidationData'], loc='upper right')
plt.show()

# Visualization of the model using tensor board
tbCallBack = TensorBoard(log_dir='./2_logistic_regression', histogram_freq=0, write_graph=True, write_images=True)

# Fitting the model defined using the training data along with validation using test data
history = nnHeart.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, verbose=1, initial_epoch=0)

# Evaluation of the loss and accuracy associated to the test data set
[test_loss, test_acc] = nnHeart.evaluate(x_test, y_test)
print("Evaluation result on Test Data using Tensorflow : Loss = {}, accuracy = {}".format(test_loss, test_acc))

# Listing all the components of data present in history
print('The data components present in history using Tensorflow are', history.history.keys())

# Graphical evaluation of accuracy associated with training and validation data
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Evaluation of Data Accuracy using Tensorflow')
plt.xlabel('epoch')
plt.ylabel('Accuracy of Data')
plt.legend(['TrainData', 'ValidationData'], loc='upper right')
plt.show()

# Graphical evaluation of loss associated with training and validation data
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('Loss of Data')
plt.title('Evaluation of Data Loss using Tensorflow')
plt.legend(['TrainData', 'ValidationData'], loc='upper right')
plt.show()
