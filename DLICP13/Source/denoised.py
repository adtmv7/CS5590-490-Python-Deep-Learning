# Import libraries
from keras.callbacks import TensorBoard
from keras.layers import Input, Dense
from keras.models import Model
from matplotlib import pyplot as plt, rcParams

# Size of our encoded representations
encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# Input placeholder
input_img = Input(shape=(784,))
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_img)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(784, activation='sigmoid')(encoded)

# Mapping input to its reconstruction
autoencoder = Model(input_img, decoded)
# Model mapping an input to its encoded representation
encoder = Model(input_img, encoded)
# Creating a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# Retrieving the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# Creating the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))
# Compiling the model defined
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['acc'])

# Loading input data set
from keras.datasets import fashion_mnist
import numpy as np
(x_train, _), (x_test, _) = fashion_mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

n_rows = x_test.shape[0]
n_cols = x_test.shape[1]
mean = 0.5
stddev = 0.3
noise = np.random.normal(mean, stddev, (n_rows, n_cols))
# creating the noisy test data by adding X_test with noise
x_test_noisy = x_test + noise

# Fitting the model defined on training data set
history = autoencoder.fit(x_train, x_train, epochs=5, batch_size=256, shuffle=True, validation_data=(x_test_noisy, x_test_noisy))

# Encode and decode some digits
# Note that we take them from the *test* set
encoded_imgs = encoder.predict(x_test_noisy)
decoded_imgs = decoder.predict(encoded_imgs)

# Evaluation of the results of the model obtained using the test data set
[test_loss, test_acc] = autoencoder.evaluate(x_test_noisy, x_test_noisy)
print("Evaluation result on Test Data : Loss = {}, accuracy = {}".format(test_loss, test_acc))

# Listing all the components of data present in history
print('The data components present in history are', history.history.keys())

# Graphical evaluation of accuracy associated with training and validation data
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Evaluation of Data Accuracy')
plt.xlabel('epoch')
plt.ylabel('Accuracy of Data')
plt.legend(['TrainData', 'ValidationData'], loc='upper right')
plt.show()

# Graphical evaluation of loss associated with training and validation data
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('Loss of Data')
plt.title('Evaluation of Data Loss')
plt.legend(['TrainData', 'ValidationData'], loc='upper right')
plt.show()

# deciding how big we want our print out to be
rcParams['figure.figsize'] = 20,20
# looping through the first 10 test images and printing  them out
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(x_test[i].reshape(28,28),  cmap='Greys')
    plt.axis('off')
plt.show()
# printing out the noisy images
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(x_test_noisy[i].reshape(28,28),  cmap='Greys')
    plt.axis('off')
plt.show()

