from keras.datasets import imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# print(train_data[0])
# print(len(train_data[0]))
# print(train_labels[1])


def decode_text_review(user_input):
    word_index = imdb.get_word_index()
    reverse_word_index = dict(
        [(value, key) for (key, value) in word_index.items()]
    )
    decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in user_input])

    return decoded_review


# print(decode_text_review(train_data[1]))

# print(train_data.shape)
# print(test_data.shape)

# //////////////////////////////////////////////////////////////////////////////////////////

import numpy as np
from keras import models
from keras import layers
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Step.1 : Preparing the data


def vectorize_sequences_one_hot_encoding(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results


x_train = vectorize_sequences_one_hot_encoding(train_data)
x_test = vectorize_sequences_one_hot_encoding(test_data)

# print(train_data[0])
# print(x_train[0])
# print(x_test)

# print(test_labels.shape)

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

# print(test_labels.shape)
# print(y_test.shape)

# Step.2 : Define Model

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# Step.3 : Compile the Model

model.compile(optimizer='rmsprop',          # rmsprop is an optimizer function already defined in Keras
              loss='binary_crossentropy',   # binary_crossentropy is a loss function already defined in Keras
              metrics=['accuracy'])       # accuracy is a metrics function already defined in Keras

# Alternatively, you can use custom functions for optimizer, loss and metrics. See page 73

# Step.4 : Validate your approach
#
# 4.1 : Set aside a validation set by setting apart 10,000 samples from the original training data

x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]

# print(x_train)
# print(x_train.shape)

# Step.5 : Fit the model

history = model.fit(    #model.fit() returns a history object - a dictionary of what happened during training
    partial_x_train,
    partial_y_train,
    epochs=20,
    batch_size=512,
    validation_data=(x_val, y_val)
)

# Use following after running until end Step.6 first and plotting & observing overfitting
# Here we reduce epoch to reduce overfitting
# We reduce epoch from 20 to 4 because we learned from plots that
# after 4th epoch, the loss on validation data starts increasing
# also after 4th epoch, the accuracy on validation data starts decreasing

# history = model.fit(
#     partial_x_train,
#     partial_y_train,
#     epochs=4,
#     batch_size=512,
#     validation_data=(x_val, y_val)
# )

history_dict = history.history
print(history_dict.keys())  # returns dict_keys(['val_loss', 'val_acc', 'loss', 'acc'])

# Step.6 : Plot : training and validation loss & training and validation accuracy

# Plotting Loss

loss_values = history_dict['loss']
validation_loss_values = history_dict['val_loss']

epochs = range(1, len(loss_values) + 1)

plt.plot(epochs, loss_values, 'bo', label='Training Loss')
plt.plot(epochs, validation_loss_values, 'b', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# Plotting Accuracy

plt.clf()  # Clear the figure

acc_values = history_dict['acc']
validation_acc_values = history_dict['val_acc']

plt.plot(epochs, acc_values, 'bo', test_labels='Training Accuracy')
plt.plot(epochs, validation_acc_values, 'b', test_labels='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# //////////////////////////////////////////////////////////////////////////////////////////

# Step.7 :

result = model.predict(x_test)
print(result)
