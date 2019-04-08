# Word Level One Hot Encoding ( Manual )

from keras.preprocessing.text import Tokenizer
import string
import numpy as np

samples = ['The cat sat on the mat', 'The dog ate my homework.']

token_index = {}

for sample in samples:
    for word in sample.split():
        if word not in token_index:
            token_index[word] = len(token_index) + 1

# print(token_index)

max_length = 10

# print(np.zeros((5,), dtype=int))
# print(np.zeros((5,), dtype=float))
# print(np.zeros((3, 3)))

results = np.zeros(shape=(len(samples),
                          max_length,
                          max(token_index.values()) + 1
                          ))

# print(results)

for i, sample in enumerate(samples):
    for j, word in list(enumerate(sample.split()))[:max_length]:
        index = token_index.get(word)
        results[i, j, index] = 1

# print(results)


# Automatic ( Keras )

# Creates a tokenizer, configured to only take into account the 1000 most common words
tokenizer = Tokenizer(num_words=1000)
# Builds the word index
tokenizer.fit_on_texts(samples)

# Turn index into list of integer indices
sequences = tokenizer.texts_to_sequences(samples)

# Get the one-hot binary representations
# Vectorization modes other than one-hot encoding are supported by tokenizer
one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')

word_index = tokenizer.word_index

# print('Found %s unique tokens.' % len(word_index))







