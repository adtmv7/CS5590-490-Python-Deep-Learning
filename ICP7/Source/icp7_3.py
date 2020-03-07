import nltk
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk import wordpunct_tokenize, pos_tag, ne_chunk
from nltk import ngrams

test_text = open('input.txt', encoding="utf8").read()

# a) Tokenization
token = nltk.word_tokenize(test_text)
print('Tokens identified are', token)


# b) Part Of Speech tagging
pos = nltk.pos_tag(token)
print('Part of Speech associated with the input text', pos)


# c) Stemming - identifying the root or base word of the terms associated
pStemmer = PorterStemmer() #Porter Stemming keeps only prefix for each words and leave non English words like troubl.
for x in token:
    print('Result of Stemming using PorterStemmer for ', x, 'is ', pStemmer.stem(x))

lStemmer = LancasterStemmer() #Lancaster stemming is a rule-based stemming based on the last letter of the words.
for y in token:
    print('Result of Stemming using LancasterStemmer for ', y, 'is ', lStemmer.stem(y))

sStemmer = SnowballStemmer('english')
for z in token:
    print('Result of Stemming using SnowballStemmer for ', z, 'is', sStemmer.stem(z))


# d) Lemmatization - noramlization of text based on the meaning as part of the speech (converts plurals or adjective to
# their basic, meaningful singular form)
lemmatizer = WordNetLemmatizer()
print('Result of Lemmatization: ', lemmatizer.lemmatize(x))

# e) Trigram
trigram = ngrams(test_text.split(), 3)
for gram in trigram:
    print('Trigram data is ', gram)
print(str(trigram))

# f) Named Entity Recognition
print('Named Entitiy Recognition is ', ne_chunk(pos_tag(wordpunct_tokenize(test_text))))