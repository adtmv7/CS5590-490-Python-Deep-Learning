import nltk
from nltk.tokenize import word_tokenize, sent_tokenize, wordpunct_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import ngrams

# a) Read the text from the file
text = open('nlp_input.txt').read()
word_tokens = nltk.word_tokenize(text)
for t in word_tokens: #Word Tokenization
    print ("Word Tokenization : \n", word_tokens)

# Lemmatization - noramlization of text based on the meaning as part of the speech
lemmatizer = WordNetLemmatizer() # creating a lemmatization object
print("Lemmatization :\n")
for x in word_tokens:
    print(lemmatizer.lemmatize(str(x)))


#Trigram for the words
trigram_output = []
trigrams=ngrams(word_tokens,3)
for y in trigrams:
  trigram_output.append(y)
  print(trigram_output)

#Extractions of Top 10 most repeated trigrams based on their count
frequency = nltk.FreqDist(trigram_output)

# Printing the most common words
commontrigram = frequency.most_common()
print("Frequency of Trigrams are : \n", commontrigram)

top10 = frequency.most_common(10) # Top 10 Trigrams
print("Top 10 Trigrams are : \n", top10)

#Getting most repeated trigram sentences using sentence tokenization.
sentencetokens = nltk.sent_tokenize(text)

#Array to append the sentences.
concatenated_output = []

#Iterating the sentense
for sentence in sentencetokens:
  for a,b,c in trigram_output: #iterating all trigrams
    for((d,e,f),length) in top10: #iterating the top 10 from all trigrams
      if(a,b,c==d,e,f): #comparing both
        concatenated_output.append(sentence)
print("Concatenated Array : ",concatenated_output)
