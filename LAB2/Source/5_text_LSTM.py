# Import libraries
import re
import pandas as pd
from keras import Sequential
from keras.callbacks import TensorBoard
from keras.constraints import maxnorm
from keras.layers import Embedding, Conv1D, Dropout, MaxPooling1D, Flatten, Dense, LSTM, SpatialDropout1D
from keras.optimizers import SGD
from keras.utils import to_categorical
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Load movie review data set
train_movie_df = pd.read_csv('train.tsv', delimiter='\t', encoding='utf-8')
test_movie_df = pd.read_csv('test.tsv', delimiter='\t', encoding='utf-8')

# cleaning the data set to identify features that are important
train_movie_df = train_movie_df.drop(columns=['PhraseId', 'SentenceId'])
test_movie_df = test_movie_df.drop(columns=['PhraseId', 'SentenceId'])

train_movie_df['Phrase'] = train_movie_df['Phrase'].apply(lambda x: re.sub('[^a-zA-z0-9\s]', '', x.lower()))
test_movie_df['Phrase'] = test_movie_df['Phrase'].apply(lambda x: re.sub('[^a-zA-z0-9\s]', '', x.lower()))

max_features = 2000
tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(train_movie_df['Phrase'].values)
X_train = tokenizer.texts_to_sequences(train_movie_df['Phrase'].values)
X_train = pad_sequences(X_train)

tokenizer.fit_on_texts(test_movie_df['Phrase'].values)
X_test = tokenizer.texts_to_sequences(test_movie_df['Phrase'].values)
X_test = pad_sequences(X_train)

# Creating the model
embed_dim = 128
lstm_out = 196

# Design the CNN using LSTM classification
# Model defined
model = Sequential()
# Input layer of the model for processing
model.add(Embedding(max_features, embed_dim, input_length=X_train.shape[1]))
# Hidden layer
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dropout(0.5))
# Output layer
model.add(Dense(5, activation='softmax'))
# Compile the model identified
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Identify the data into training and test sets
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(train_movie_df['Sentiment'])
Y_train = to_categorical(integer_encoded)
x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.3, random_state=60)

# Fitting the model identified using the training data set
history = model.fit(x_train, y_train, epochs=2, batch_size=500, validation_data=(x_test, y_test))

# Evaluation of the results of the model obtained using the test data set
[test_loss, test_acc] = model.evaluate(x_test, y_test)
print("Evaluation result on Test Data : Loss = {}, accuracy = {}".format(test_loss, test_acc))

# Listing all the components of data present in history
print('The data components present in history are', history.history.keys())

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
tbCallBack = TensorBoard(log_dir='./lab2_1', histogram_freq=0, write_graph=True, write_images=True)

# Fitting the model defined using the training data along with validation using test data
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=2, verbose=1, initial_epoch=0)

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