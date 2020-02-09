import numpy as np

#Generate random integers
x = np.random.randint(1, 20, 15)
print("List of randomly generated integers: ", x)

#Reshaping randomly generated integers into 3X5 Matrix
a = x.reshape(3, 5)
print("Reshape generated arrays into 3X5 Matrix")
print(a)
a[np.where(a==np.max(a))] = 0
print("Maximum value replaced by 0:")
print(a)