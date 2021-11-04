import cv2
import numpy as np
from scipy import misc
i = misc.ascent()

##
import matplotlib.pyplot as plt
plt.grid(False)
plt.gray()
plt.axis('off')
plt.imshow(i)
plt.show()
##
i_transformed = np.copy(i)
size_x = i_transformed.shape[0]
size_y = i_transformed.shape[1]
##
# This filter detects edges nicely
# It creates a filter that only passes through sharp edges and straight lines.
# Experiment with different values for fun effects.
#filter = [ [0, 1, 0], [1, -4, 1], [0, 1, 0]]
# A couple more filters to try for fun!
filter = [ [-1, -2, -1], [0, 0, 0], [1, 2, 1]]
#filter = [ [-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
 # If all the digits in the filter don't add up to 0 or 1, you
# should probably do a weight to get it to do so
# so, for example, if your weights are 1,1,1 1,2,1 1,1,1
# They add up to 10, so you would set a weight of .1 if you want to normalize them
weight  = 1
##
for x in range(1,size_x-1):
  for y in range(1,size_y-1):
      output_pixel = 0.0
      output_pixel = output_pixel + (i[x - 1, y-1] * filter[0][0])
      output_pixel = output_pixel + (i[x, y-1] * filter[0][1])
      output_pixel = output_pixel + (i[x + 1, y-1] * filter[0][2])
      output_pixel = output_pixel + (i[x-1, y] * filter[1][0])
      output_pixel = output_pixel + (i[x, y] * filter[1][1])
      output_pixel = output_pixel + (i[x+1, y] * filter[1][2])
      output_pixel = output_pixel + (i[x-1, y+1] * filter[2][0])
      output_pixel = output_pixel + (i[x, y+1] * filter[2][1])
      output_pixel = output_pixel + (i[x+1, y+1] * filter[2][2])
      output_pixel = output_pixel * weight
      if(output_pixel<0):
        output_pixel=0
      if(output_pixel>255):
        output_pixel=255
      i_transformed[x, y] = output_pixel
##
# Plot the image. Note the size of the axes -- they are 512 by 512
plt.gray()
plt.grid(False)
plt.imshow(i_transformed)
#plt.axis('off')
plt.show()
##
new_x = int(size_x / 2)
new_y = int(size_y / 2)
newImage = np.zeros((new_x, new_y))
for x in range(0, size_x, 2):
    for y in range(0, size_y, 2):
        pixels = []
        pixels.append(i_transformed[x, y])
        pixels.append(i_transformed[x + 1, y])
        pixels.append(i_transformed[x, y + 1])
        pixels.append(i_transformed[x + 1, y + 1])
        pixels.sort(reverse=True)
        newImage[int(x / 2), int(y / 2)] = pixels[0]

# Plot the image. Note the size of the axes -- now 256 pixels instead of 512
plt.gray()
plt.grid(False)
plt.imshow(newImage)
# plt.axis('off')
plt.show()
##
####################################################
##
import tensorflow as tf
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
print(tf.shape(training_images))
print(tf.shape(training_labels))
training_images=training_images/255.0
test_images=test_images/255.0
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(training_images, training_labels, epochs=5)
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print ('Test loss: {}, Test accuracy: {}'.format(test_loss, test_accuracy*100))
##
import tensorflow as tf
print(tf.__version__)
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
training_images=training_images.reshape(60000, 28, 28, 1)
training_images=training_images / 255.0
test_images = test_images.reshape(10000, 28, 28, 1)
test_images=test_images / 255.0
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(training_images, training_labels, epochs=5)
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print ('Test loss: {}, Test accuracy: {}'.format(test_loss, test_accuracy*100))
##
# The first step is to gather the data.
#
# You'll notice that there's a change here and the training data needed to be reshaped.
# That's because the first convolution expects a single tensor containing everything, ' \
#     'so instead of 60,000 28x28x1 items in a list, you have a single 4D list that is ' \
#     '60,000x28x28x1, and the same for the test images. If you don't do that, then
# you'll get an error when training because the convolutions do not recognize the shape.
#
#
# import tensorflow as tf
# mnist = tf.keras.datasets.fashion_mnist
# (training_images, training_labels), (test_images, test_labels) = mnist.load_data()
# training_images=training_images.reshape(60000, 28, 28, 1)
# training_images = training_images/255.0
# test_images = test_images.reshape(10000, 28, 28, 1)
# test_images = test_images/255.0
#

# Next, define your model. Instead of the input layer at the top, you're going to add a
# convolutional layer. The parameters are:
#
# The number of convolutions you want to generate. A value like 32 is a good starting point.
# The size of the convolutional matrix, in this case a 3x3 grid.
# The activation function to use, in this case use relu.
# In the first layer, the shape of the input data.
# You'll follow the convolution with a max pooling layer, which is designed to compress the
# image while maintaining the content of the features that were highlighted by the convolution.
# By specifying (2,2) for the max pooling, the effect is to reduce the size of the image by a
# factor of 4. It creates a 2x2 array of pixels and picks the largest pixel value, turning 4 pixels
# into 1. It repeats this computation across the image, and in so doing halves the number of horizontal
# pixels and halves the number of vertical pixels.
#
# You can call model.summary() to see the size and shape of the network. Notice that after every max
# pooling layer, the image size is reduced in the following way:
#
#
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# conv2d_2 (Conv2D)            (None, 26, 26, 64)        640
# _________________________________________________________________
# max_pooling2d_2 (MaxPooling2 (None, 13, 13, 64)        0
# _________________________________________________________________
# conv2d_3 (Conv2D)            (None, 11, 11, 64)        36928
# _________________________________________________________________
# max_pooling2d_3 (MaxPooling2 (None, 5, 5, 64)          0
# _________________________________________________________________
# flatten_2 (Flatten)          (None, 1600)              0
# _________________________________________________________________
# dense_4 (Dense)              (None, 128)               204928
# _________________________________________________________________
# dense_5 (Dense)              (None, 10)                1290
# =================================================================
# Here's the full code for the CNN:
#
#
# model = tf.keras.models.Sequential([
# tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
# tf.keras.layers.MaxPooling2D(2, 2),
# #Add another convolution
# tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
# tf.keras.layers.MaxPooling2D(2, 2),
# #Now flatten the output. After this you'll just have the same DNN structure as the non convolutional version
# tf.keras.layers.Flatten(),
# #The same 128 dense layers, and 10 output layers as in the pre-convolution example:
# tf.keras.layers.Dense(128, activation='relu'),
# tf.keras.layers.Dense(10, activation='softmax')
# ])
##
# Compile the model, call the fit method to do the training, and evaluate the loss and accuracy from the test set.
#
#
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# model.fit(training_images, training_labels, epochs=5)
# test_loss, test_acc = model.evaluate(test_images, test_labels)
# print ('Test loss: {}, Test accuracy: {}'.format(test_loss, test_acc*100))
##
import matplotlib.pyplot as plt
f, axarr = plt.subplots(3,4)
FIRST_IMAGE=0
SECOND_IMAGE=23
THIRD_IMAGE=28
CONVOLUTION_NUMBER = 6
from tensorflow.keras import models
layer_outputs = [layer.output for layer in model.layers]
activation_model = tf.keras.models.Model(inputs = model.input, outputs = layer_outputs)
for x in range(0,4):
  f1 = activation_model.predict(test_images[FIRST_IMAGE].reshape(1, 28, 28, 1))[x]
  axarr[0,x].imshow(f1[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
  axarr[0,x].grid(False)
  f2 = activation_model.predict(test_images[SECOND_IMAGE].reshape(1, 28, 28, 1))[x]
  axarr[1,x].imshow(f2[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
  axarr[1,x].grid(False)
  f3 = activation_model.predict(test_images[THIRD_IMAGE].reshape(1, 28, 28, 1))[x]
  axarr[2,x].imshow(f3[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
  axarr[2,x].grid(False)