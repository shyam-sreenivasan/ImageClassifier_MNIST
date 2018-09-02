# MNIST data image classfier with keras

#load the dataset
from keras.datasets import mnist 
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

from keras import models
from keras import layers

#create a sequential model
network = models.Sequential()
# layer one with 512 nodes (arbitrary) and relu as activation function, input_shape is the input dimension expected by the layer
# here, it is 28 by 28 pixel, flattened to 784 x 1 vector
network.add(layers.Dense(512, activation='relu',input_shape=(28*28,)))
#output layer 
network.add(layers.Dense(10, activation='softmax'))

#compile the network by providing the optimizer, loss function and the metrics to monitor
network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

#flatten out the input data into a 784x1 vector, convert pixel values from uint8 to float32, and range from 
# 0 255 to the interval [0,1]
train_images = train_images.reshape((60000,28*28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000,28*28))
test_images = test_images.astype('float32') / 255

#convert the labels to one hot vector form
#Example , if there are 3 classes 1,2 and 3, the one hot vector representation would be like
# class 1 => [1, 0, 0]
# class 2 => [0, 1, 0]
# class 3 => [0, 0, 1]
from keras.utils import to_categorical
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

#epochs is the number of iterations over the entire dataset
network.fit(train_images, train_labels, epochs=5, batch_size=128)

# Evaluate the model on the test data.
test_loss, test_acc = network.evaluate(test_images, test_labels)

print('test accuracy: ', test_acc)
