# Importing requests library to send HTTP requests
# Parsing data using BeautifulSoup function
import requests
from bs4 import BeautifulSoup

#Parsing the webpage
webpage = "https://en.wikipedia.org/wiki/Deep_learning"
Parsedpage = requests.get(webpage).text
soup = BeautifulSoup(Parsedpage,"html.parser")

# Print the title of the web page
title = soup.title
print(title)

# Finding all the links within the page containing a tag
Tag = soup.find_all("a")
for link in Tag:
     print(link.get("href"))