# Import libraries
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re
from sklearn.preprocessing import LabelEncoder

# Read the input data from csv file
data = pd.read_csv('spam.csv', encoding='latin1')

# cleaning the data set to identify features that are important
data = data[['v1','v2']]

data['v2'] = data['v2'].apply(lambda x: x.lower())
data['v2'] = data['v2'].apply((lambda x: re.sub('[^a-zA-z0-9\s]', '', x)))

for idx, row in data.iterrows():
    row[0] = row[0].replace('rt', ' ')

max_features = 2000
tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(data['v2'].values)
X = tokenizer.texts_to_sequences(data['v2'].values)

X = pad_sequences(X)

# Creating the model to be fit
embed_dim = 128
lstm_out = 196
# Define model used along with the appropriate layers
def createmodel():
    model = Sequential()
    model.add(Embedding(max_features, embed_dim,input_length = X.shape[1]))
    model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(2,activation='softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
    return model
# print(model.summary())

# Identify the data into training and test sets
labelencoder = LabelEncoder()
integer_encoded = labelencoder.fit_transform(data['v1'])
y = to_categorical(integer_encoded)
X_train, X_test, Y_train, Y_test = train_test_split(X,y, test_size = 0.33, random_state = 42)

# Fitting the training data on the model defined
batch_size = 32
model = createmodel()
model.fit(X_train, Y_train, epochs = 10, batch_size=batch_size, verbose = 2)

# Evaluation of the performance of the model fit
score, acc = model.evaluate(X_test, Y_test, verbose=2, batch_size=batch_size)
print('The score obtained from the model fit is ', score)
print('The accuracy of the model fit is ', acc)
print(model.metrics_names)