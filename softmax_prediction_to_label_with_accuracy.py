import tensorflow as tf
print(tf.__version__)
##
mnist = tf.keras.datasets.fashion_mnist
##
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
##
import matplotlib.pyplot as plt
plt.imshow(training_images[0])
print(training_labels[0])
print(training_images[0])
##
training_images  = training_images / 255.0
test_images = test_images / 255.0
##
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
##
model.compile(optimizer = tf.keras.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=5)
##!!!!!!!!!!!!!The key to change from softmax decimal values to prediction labels and find the accuracy for this test set.
import numpy as np
from sklearn.metrics import accuracy_score
raw = model.predict(test_images)
y_hat = np.zeros(len(raw))
for i in range(len(raw)):
  y_hat[i] = np.argmax(raw[i])
accuracy_score(test_labels, y_hat)