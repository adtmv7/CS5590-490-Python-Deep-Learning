# Importing libraries
from sklearn.datasets import fetch_20newsgroups
from keras.models import Sequential
from keras import layers
from keras.preprocessing.text import Tokenizer
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard

# Reading input data of IMBD reviews
newsgroups_train = fetch_20newsgroups(subset='train', shuffle=True)
sentences = newsgroups_train.data
y = newsgroups_train.target

# Identifying the number of input dimensions present within sentences to build the network model
max_review_len = max([len(s.split()) for s in sentences])
print('Maximum length of review is', max_review_len)

# Tokenizing data
tokenizer = Tokenizer(num_words=max_review_len)
tokenizer.fit_on_texts(sentences)

# Getting the vocabulary of data
sentences = tokenizer.texts_to_matrix(sentences)

le = preprocessing.LabelEncoder()
y = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(sentences, y, test_size=0.25, random_state=1000)

# Model defined and fit
model = Sequential()

# Input layer of model defined
model.add(layers.Dense(300, input_dim=max_review_len, activation='relu'))
# Hidden layer of model defined
model.add(layers.Dense(20, activation='softmax'))
# Output layer of model defined
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])

# Model fit using training data set
history = model.fit(X_train, y_train, epochs=5, verbose=True, validation_data=(X_test, y_test), batch_size=256)

# Evaluation of the performance of the model fit
[test_loss, test_acc] = model.evaluate(X_test, y_test)
print("Evaluation result on Test Data : Loss = {}, accuracy = {}".format(test_loss, test_acc))

# Graphical evaluation of accuracy associated with training and validation data
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
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

# Tensor board analysis
tensorAnalysis = TensorBoard(log_dir="logs", histogram_freq=1, write_graph=True, write_images=False)
history = model.fit(X_train, y_train, verbose=1, validation_data=(X_test, y_test), callbacks=[tensorAnalysis])

# Evaluation of the performance of the model fit using Tensorflow
[test_loss, test_acc] = model.evaluate(X_test, y_test)
print("Evaluation result using Tensorflow on Test Data : Loss = {}, accuracy = {}".format(test_loss, test_acc))

# Graphical evaluation of accuracy associated with training and validation data
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
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
