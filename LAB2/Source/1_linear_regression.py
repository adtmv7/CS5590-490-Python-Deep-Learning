# Importing libraries
import pandas as pd
from keras.callbacks import TensorBoard
from keras.optimizers import Adam, SGD
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot as plt

# Reading the dataset
deathrate = pd.read_csv('DeathRate.csv')

# Identifying features and predictor variables associated with the data
x = deathrate.iloc[:, 0:13]
y = deathrate.iloc[:, 13]

# Split the data set into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=100)
print(x_train.size)
print(x_train.shape)

# Hyper parameters set 1
# activation_function='relu'
# learning_rate=0.1
# epochs=100
# b_size=50
# decay_rate= learning_rate / epochs
# optimizer = Adam(lr=learning_rate, decay=decay_rate)

# Hyper parameters set 2
activation_function="tanh"
learning_rate=0.5
epochs=150
b_size=75
decay_rate= learning_rate / epochs
optimizer = SGD(lr=learning_rate, decay=decay_rate)

# Creating neural network to perform linear regression
# Model Creation
model = Sequential()
# Providing inputs to the hidden layer
model.add(Dense(100, input_dim=13, activation=activation_function))
# Adding hidden layers and activation functions
model.add(Dense(15, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(255, activation='tanh'))
# output layer
model.add(Dense(1, activation='sigmoid'))
# Compiling the defined model
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Fitting the defined model using the training data set
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100, batch_size=b_size, verbose=0, initial_epoch=0)

# Evaluation of the loss and accuracy associated to the test data set
[test_loss, test_acc] = model.evaluate(x_test, y_test)
print("Evaluation result on Test Data : Loss = {}, accuracy = {}".format(test_loss, test_acc))

# Listing components of data present in history
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
tbCallBack = TensorBoard(log_dir='./q1_linear_regression', histogram_freq=0, write_graph=True, write_images=True)

# Fitting the model defined using the training data along with validation using test data
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, verbose=1, initial_epoch=0)

# Evaluation of the loss and accuracy associated to the test data set
[test_loss, test_acc] = model.evaluate(x_test, y_test)
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


