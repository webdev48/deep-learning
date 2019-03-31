from keras import layers
from keras import models
from keras.datasets import mnist
from keras.utils import to_categorical

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# If you want to calculate number of parameter at each layer:
# https://stackoverflow.com/questions/50158474/how-to-calculate-the-total-number-of-parameters-in-a-convolutional-neural-networ

# Also, try to understand the meanings on this:
# https://stackoverflow.com/questions/42786717/how-to-calculate-the-number-of-parameters-for-convolutional-neural-network


# Number of Channels = (channels_in * kernel_width * kernel_height * channels_out) + num_channels
# Params For Conv 2D Layer 1 = (1x3x3x32) + 32 = 320
# Params For Conv 2D Layer 2 = (32x3x3x64) + 64 = 18,496
# Params For Conv 2D Layer 3 = (64x3x3x64) + 64 = 36,928

# The next step is to feed the last output tensor (of shape (3, 3, 64)) into a
# densely connected classifier network. These classifiers process vectors, which are 1D,
# whereas the current output is a 3D tensor. First we have to flatten the 3D outputs to 1D

model.add(layers.Flatten())
# the (3, 3, 64) outputs are flattened into vectors of shape (576,) before going through two Dense layers.
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# Params For Dense Layer 1 = (64x3x3x64) + 64 = 36,928
# Params For Dense Layer 2 = (64+1) * 10 = 650 ( HOW IS THI CALCULATED ?? )

# Load & Reshape Data in the Model :

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print('Before reshaping:')
print(train_images.shape)
print(train_images[0])
train_images = train_images.reshape((60000, 28, 28, 1))
print('After reshaping:')
print(train_images.shape)
print(train_images[0])

train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 25

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Compile the Model

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Fit the Model

model.fit(train_images, train_labels, epochs=5, batch_size=64)

# Evaluate the Model

test_loss, test_acc = model.evaluate(test_images, test_labels)

print('test loss: ')
print(test_loss)
print('test accuracy: ')
print(test_acc)

# test loss:
# 0.2131026353994232
# test accuracy:
# 0.9832




