# Import

from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical

# Learn and explore data in hand
( train_images, train_labels) , (test_images, test_labels ) = mnist.load_data()

print( train_images.shape )
print( train_labels )

print( test_images.shape )
print( test_labels )

# Create Network
network = models.Sequential()

# All Layers to Network
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add( layers.Dense( 10, activation='softmax'))

# Configure Network ( optimizer and loss function )
network.compile( optimizer='rmsprop',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])

# Pre process Data


# In the case of mnist ,
#   transforming 60,000 28x28 ( each having value between 0 and 255 ) - uint8
#   to 60,000 28 x 28 ( each having value between 0 and 1 ) - float32

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255


# Prepare labels ( one hot encoding ? ) : LEARN

train_labels = to_categorical( train_labels )
test_labels = to_categorical( test_labels )

# Train the network ( call .fit() )

network.fit( train_images, train_labels, epochs = 5 , batch_size = 128 )

# Test the model

test_loss, test_acc = network.evaluate( test_images, test_labels)
print( 'test accuracy ', test_acc )





