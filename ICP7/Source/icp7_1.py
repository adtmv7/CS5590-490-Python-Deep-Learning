import requests
from bs4 import BeautifulSoup

# URL from which data is to be pulled
webpage = requests.get('https://en.wikipedia.org/wiki/Google').text

# Parsing data using BeautifulSoup function
soup = BeautifulSoup(webpage, 'html.parser')

# Displaying the web page text
print('Webpage Text: ', soup.title.string)