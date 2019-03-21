from keras.datasets import reuters
import numpy as np
from keras.utils.np_utils import to_categorical
from keras import models
from keras import layers
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

print(len(train_data))
print(len(test_data))
print('\n')

# decoding news wire string ( see page : 78/79 )

# Step 1. Preparing the data :

# Vectorize Input Features

def vectorize_sequences(sequence, dimension=10000):
    results = np.zeros((len(sequence), dimension))
    for i, sequence in enumerate(sequence):
        results[i, sequence] = 1
    return results


x_train = vectorize_sequences(train_data)
y_train = vectorize_sequences(test_data)

# print(train_data[10])
# print(x_train[10])
# print(train_labels[10])
# print(y_train[10])

# Vectorize Labels

# One hot encoding is widely used for categorical data,
# also known as categorical encoding

def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1
    return results


# print('Before one hot encoding : train label')
# print(train_labels.shape)
# print(train_labels[0])

one_hot_train_labels = to_one_hot(train_labels)

# print('After one hot encoding : train label')
# print(one_hot_train_labels[0])

one_hot_test_labels = to_one_hot(test_labels)

# Alternative way to do one-hot encoding ( built in method in Keras )
# is to use to_categorical function

# print(train_labels[0])
# one_hot_train_labels = to_categorical(train_labels)
# print(one_hot_train_labels[0])
# one_hot_test_labels = to_categorical(test_labels)


# Step 2. Build Network :

model = models.Sequential()

# For IMDB we only had 16 node layers because the dimensionality of the o/p space was only 2 ( + or - )
# In this example, the dimensionality of the output space is 46
# If we chose layers less than 46, layers will have information bottleneck
# Information bottleneck is when layers drop some information relevant to classification problem
# in hand which can never be recovered back. Hence, 64 node in each layer
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))

# 46 mutually exclusive topic - multi class classification
# In IMDB example, we have binary class classification where we only had 1 o/p class
# Here we have 46 output, each will encode a different output class
# This layer uses 'softmax' activation, which means the network will o/p PROBABLITY DISTRIBUTION
# over the 46 dimensional output where output[i] is the probability thar the sample belongs to class i
# The 46 node scores will sum to 1
model.add(layers.Dense(46, activation='softmax'))

# Step 3. Compile and Create the Network:

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 'categorical_crossentropy' measures the distance between two probablity distributions


x_val = x_train[:1000]
partial_x_train = x_train[1000:]

y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

# history = model.fit(
#     partial_x_train,
#     partial_y_train,
#     epochs=20,
#     batch_size=512,
#     validation_data=(x_val, y_val)
# )

# Use 9 epochs only after evaluating 20 epochs:

history = model.fit(
    partial_x_train,
    partial_y_train,
    epochs=9,
    batch_size=512,
    validation_data=(x_val, y_val)
)


# Step 4. Plot Results, Evaluate and Adjust:

loss = history.history['loss']
validation_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs, validation_loss, 'b', label='Validation Loss')

plt.title('Training and validation loss')

plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.legend()

plt.show()

plt.clf()

acc = history.history['acc']
validation_acc = history.history['val_acc']

plt.plot(epochs, acc, 'bo', label='Training Accuracy')
plt.plot(epochs, validation_acc, 'b', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# After above experiemnt, seems like the validation loss is not decreasing after 9 epochs
# and also training accuracy starts to peak at 9 epochs ( overfitting )

# HENCE retrain the network for 9 epochs

# Step 5. Evaluate final network on test set :

results = model.evaluate(y_train, one_hot_test_labels)
print('\nFinal Results:\n')
print(model.metrics_names)
print(results)

# Final Results:
# ['loss', 'acc']
# [0.9810770789322212, 0.7880676759212865]

# Step 6. Predict:

predictions = model.predict(y_train)


# Shape of each prediction should be distribution over 46 nodes

print(predictions[0].shape)

# Sum of all nodes for each prediction should sum to 1

print(np.sum(predictions[0]))

# Check for highest probablity value node for each prediction for first 3 results

print(np.argmax(predictions[0]))
print(np.argmax(predictions[1]))
print(np.argmax(predictions[2]))

# More:

# Experiment#1 :

# Do the experiment of NOT using 1 hot encoding for labels so using there integar representation
# with 'sparse_categorical_crossentropy'

# Experiment#2 :

# All other experiments on page.84:

# Decreasing nodes significantly in one of the intermediate hidden layers
# Increasing or decreasing the number of hidden layers

