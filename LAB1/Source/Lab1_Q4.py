#4)	Go to https://catalog.umkc.edu/course-offerings/graduate/comp-sci/ and fetch the course name and overview of
# course. Hint:Use BeautifulSoup package.

from bs4 import BeautifulSoup
import requests

# Enter URL to fetch the data
url = requests.get("https://catalog.umkc.edu/course-offerings/graduate/comp-sci/")
data = url.text
output = BeautifulSoup(data, "html.parser")

# Using FindAll to find desired class
for p in output.findAll('p',{'class':'courseblocktitle'}):
    for title, detail in zip(output.findAll('span', {'class': 'title'}),
                             output.findAll('p', {'class': 'courseblockdesc'})):

        # #printing courese name and course description
        print("Course Name: ", title.string)
        print("Course Description: ", detail.string)
        print('\n')
