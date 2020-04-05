
# Importing libraries
from keras import Sequential
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input

# Loading input data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Display the second image in the training data
plt.imshow(train_images[1, :, :], cmap='gray')
plt.title('Ground Truth : {}'.format(train_labels[1]))
plt.show()

# Process the data
# 1. Convert each image of shape 28*28 to 784 dimensional which will be fed to the network as a single feature
dimData = np.prod(train_images.shape[1:])
train_data = train_images.reshape(train_images.shape[0], dimData)
test_data = test_images.reshape(test_images.shape[0], dimData)

# Convert data to float and scale values between 0 and 1
train_data = train_data.astype('float')
test_data = test_data.astype('float')

# Scale data
train_data /=255.0
test_data /=255.0

# Change the labels from integer to one-hot encoding
train_labels_one_hot = to_categorical(train_labels)
test_labels_one_hot = to_categorical(test_labels)

# Creating network
a=Input(shape=(dimData,))
b= Dense(512, activation='relu')(a)
b= Dense(512, activation='relu')(b)
b= Dense(10, activation='softmax')(b)
model = Model(input=a,output=b)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_data, train_labels_one_hot, batch_size=256, epochs=2, verbose=1,
                   validation_data=(test_data, test_labels_one_hot))


