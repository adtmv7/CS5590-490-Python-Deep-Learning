#Word count in file

file = open('icp2file.txt','r')
Count= dict()                               #create new dictionary
for line in file:                           # Loop through each line of the file
    words= line.strip().split(" ")          #Split the line into words
    for word in words:
        if word in Count:                   #Check word in dictionary is present
            Count[word] = Count[word]+1
        else:
            Count[word] = 1
print (Count)
output= open("icp2file.txt",'a')            #Storing output to the same file"
output.write(str(Count))