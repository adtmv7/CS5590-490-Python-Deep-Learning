# Importing libraries
from keras.models import Sequential
from keras import layers
from keras.preprocessing.text import Tokenizer
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences


# Reading input data of IMBD reviews
df = pd.read_csv('imdb_master.csv',encoding='latin-1')
print(df.head())
sentences = df['review'].values
y = df['label'].values

# Identifying the number of input dimensions present within sentences to build the network model
max_len_review = max([len(s.split()) for s in sentences])
print('Maximum length of review is', max_len_review)

# Tokenizing data
tokenizer = Tokenizer(num_words=max_len_review)
tokenizer.fit_on_texts(sentences)

# Getting the vocabulary of data
sentences = tokenizer.texts_to_matrix(sentences)

# Encoding target associated with IMDB data of reviews
le = preprocessing.LabelEncoder()
y = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(sentences, y, test_size=0.25, random_state=1000)

# Model defined and fit
model = Sequential()

# Input layer of model defined
model.add(layers.Dense(300, input_dim=max_len_review, activation='relu'))
# Hidden layer of model defined
model.add(layers.Dense(3, activation='softmax'))
# Output layer of model defined
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])

# Model fit using training data set
history = model.fit(X_train, y_train, epochs=1, verbose=True, validation_data=(X_test, y_test), batch_size=256)
