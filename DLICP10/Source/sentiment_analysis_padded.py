# Importing libraries
from keras.layers import Embedding
from keras.layers import Flatten
from keras.models import Sequential
from keras import layers
from keras.preprocessing.text import Tokenizer
import pandas as pd
from keras_preprocessing.sequence import pad_sequences
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# Reading input data of IMBD reviews
df = pd.read_csv('imdb_master.csv', encoding='latin-1')
print(df.head())
sentences = df['review'].values
y = df['label'].values

# Identifying the number of input dimensions present within sentences to build the network model
max_review_len = max([len(s.split()) for s in sentences])
print('Maximum length of review is', max_review_len)

# Tokenizing data
tokenizer = Tokenizer(num_words=max_review_len)
tokenizer.fit_on_texts(sentences)

# Preparation of data for embedding layer
vocab_size = len(tokenizer.word_index)+1
sentences = tokenizer.texts_to_matrix(sentences)
padded_docs = pad_sequences(sentences, maxlen=max_review_len)

# Encoding target associated with IMDB data of reviews
le = preprocessing.LabelEncoder()
y = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(padded_docs, y, test_size=0.25, random_state=1000)

# Model defined and fit
model = Sequential()

# Adding the embedding layer to the model defined
model.add(Embedding(vocab_size, 50, input_length=max_review_len))
model.add(Flatten())
# Input layer of model defined
model.add(layers.Dense(300, input_dim=max_review_len, activation='relu'))
model.add(layers.Dense(3, activation='softmax'))
# Output layer of model defined
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])

# Model fit using training data set
history = model.fit(X_train, y_train, epochs=3, verbose=True, validation_data=(X_test, y_test), batch_size=256)

# Evaluation of the performance of the model fit
[test_loss, test_acc] = model.evaluate(X_test, y_test)
print("Evaluation result on Test Data : Loss = {}, accuracy = {}".format(test_loss, test_acc))
