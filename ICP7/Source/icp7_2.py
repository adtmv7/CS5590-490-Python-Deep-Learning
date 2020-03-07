import requests
from bs4 import BeautifulSoup

# URL from which data is to be pulled
webpage = requests.get('https://en.wikipedia.org/wiki/Google').text

# Parsing data using BeautifulSoup function
soup = BeautifulSoup(webpage, 'html.parser')

# Saving the parsed webpage data into the text file titled 'input'
text = soup.get_text()
f = open('input.txt', 'w',encoding='utf-8')
f.write(text)