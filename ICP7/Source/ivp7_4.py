# %%
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from nltk.corpus import stopwords

# %%
twenty_train = fetch_20newsgroups(subset='train', shuffle=True)

# %%
tfidf_Vect = TfidfVectorizer()
X_train_tfidf = tfidf_Vect.fit_transform(twenty_train.data)

# print(tfidf_Vect.vocabulary_)
clf = MultinomialNB()
clf.fit(X_train_tfidf, twenty_train.target)

# %%
twenty_test = fetch_20newsgroups(subset='test', shuffle=True)
X_test_tfidf = tfidf_Vect.transform(twenty_test.data)

# %%
predicted = clf.predict(X_test_tfidf)

# %%
score = metrics.accuracy_score(twenty_test.target, predicted)
print(score)

# Creating matrix from the data available (provided in lecture)
countVec = CountVectorizer()
x_train = countVec.fit_transform(twenty_train.data)
x_test = countVec.transform(twenty_test.data)

# Multinomial NB Model building on training data
clfnew = MultinomialNB()
clfnew.fit(x_train, twenty_train.target)

# Evaluating the model fit on test data set
clfPredcit = clfnew.predict(x_test)

# Computing the score of the Multinomial NB model
clfScore = metrics.accuracy_score(twenty_test.target, clfPredcit)
print('The score for Multinomial NB model is', clfScore)

# a) Building K-Nearest Neighbours Model on training data set
knnModel = KNeighborsClassifier()
knnModel.fit(x_train, twenty_train.target)

# Predicting the model fit on test data set
knnPredict = knnModel.predict(x_test)

# Computing the score of the KNN model
knnScore = metrics.accuracy_score(twenty_test.target, knnPredict)
print('The score for KNN model is', knnScore)

# b) TFIDF vectorization updated to use bigrams as asked in the question
bigramnewVec = TfidfVectorizer(ngram_range=(1, 2))
x_train_bigram = bigramnewVec.fit_transform(twenty_train.data)
x_test_bigram = bigramnewVec.transform(twenty_test.data)

# Revised models and their accuracy using bigram vectorization of data
# Multinomial NB
clfBigram = clfnew.fit(x_train_bigram, twenty_train.target)
clfPredcitBigram = clfnew.predict(x_test_bigram)
clfScoreBigram = metrics.accuracy_score(twenty_test.target, clfPredcitBigram)
print('The score for Multinomial NB model after TFIDF vectorization bigram update is', clfScoreBigram)

# K-Nearest Neighbours
knnBigram = knnModel.fit(x_train_bigram, twenty_train.target)
knnPredictBigram = knnModel.predict(x_test_bigram)
knnScoreBigram = metrics.accuracy_score(twenty_test.target, knnPredictBigram)
print('The score for KNN model is', knnScoreBigram)

# Stop word
# TFIDF vectorization updated to accommodate for stop word 'english'
stopwordVec = TfidfVectorizer(stop_words='english')
x_train_stopWord = stopwordVec.fit_transform(twenty_train.data)
x_test_stopWord = stopwordVec.transform(twenty_test.data)

# Revised models and their accuracy when stopword english is applied on data
# Multinomial NB
clfStopWord = clfnew.fit(x_train_stopWord, twenty_train.target)
clfPredcitStopWord = clfnew.predict(x_test_stopWord)
clfScoreStopWord = metrics.accuracy_score(twenty_test.target, clfPredcitStopWord)
print('The score for Multinomial NB model after stopword english update is', clfScoreStopWord)

# K-Nearest Neighbours
knnStopWord = knnModel.fit(x_train_stopWord, twenty_train.target)
knnPredictStopWord = knnModel.predict(x_test_stopWord)
knnScoreStopWord = metrics.accuracy_score(twenty_test.target, knnPredictStopWord)
print('The score for KNN model is after stop word english is updated is', knnScoreStopWord)