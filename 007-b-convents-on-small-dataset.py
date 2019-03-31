# www.kaggle.com/c/dogs-vs-cats/data
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import os, shutil

# Setup Training, Validation and Test Dataset from Original Dataset

original_dataset_dir = '/Users/anasrazafirdousi/Documents/workspace/ML/dogs-vs-cats/train'

base_dir = '/Users/anasrazafirdousi/Documents/workspace/ML/dogs-vs-cats/small'
os.mkdir(base_dir)

train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)

validation_dir = os.path.join(base_dir, 'validation')
os.mkdir(validation_dir)

test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)

train_cats_dir = os.path.join(train_dir, 'cats')
os.mkdir(train_cats_dir)

train_dogs_dir = os.path.join(train_dir, 'dogs')
os.mkdir(train_dogs_dir)

validation_cats_dir = os.path.join(validation_dir, 'cats')
os.mkdir(validation_cats_dir)

validation_dogs_dir = os.path.join(validation_dir, 'dogs')
os.mkdir(validation_dogs_dir)

test_cats_dir = os.path.join(test_dir, 'cats')
os.mkdir(test_cats_dir)

test_dogs_dir = os.path.join(test_dir, 'dogs')
os.mkdir(test_dogs_dir)


fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_cats_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_cats_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_cats_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_dogs_dir, fname)
    shutil.copyfile(src, dst)
fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_dogs_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_dogs_dir, fname)
    shutil.copyfile(src, dst)

# Sanity Check we divided and copied the imags properly

print('total training cat images:', len(os.listdir(train_cats_dir)))
print('total training dog images:', len(os.listdir(train_dogs_dir)))
print('total validation cat images:', len(os.listdir(validation_cats_dir)))
print('total validation dog images:', len(os.listdir(validation_dogs_dir)))
print('total test cat images:', len(os.listdir(test_cats_dir)))
print('total test dog images:', len(os.listdir(test_dogs_dir)))

# Build Model

model = models.Sequential()
models.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
models.add(layers.MaxPooling2D(2, 2))
models.add(layers.Conv2D(64, (3, 3), activation='relu'))
models.add(layers.MaxPooling2D(2, 2))
models.add(layers.Conv2D(128, (3, 3), activation='relu'))
models.add(layers.MaxPooling2D(2, 2))
models.add(layers.Conv2D(128, (3, 3), activation='relu'))
models.add(layers.MaxPooling2D(2, 2))
models.add(layers.Flatten())
models.add(layers.Dense(512, activation='relu'))
models.add(layers.Dense(1, activation='sigmoid'))

# Compile Network

# RMSprop optimizer, as usual. Because you ended the network with a single sigmoid unit,
#  you’ll use binary crossentropy as the loss
models.compile(loss='binary_crossentropy',
               optimizer= optimizers.RMSprop(lr=1e-4),
               metrics=['acc'])

# Data Pre Processing:
# Converting Images to Floating Point tensors

# 1. Read the picture files.
# 2. Decode the JPEG content to RGB grids of pixels.
# 3. Convert these into floating-point tensors.
# 4. Rescale the pixel values (between 0 and 255) to the [0, 1] interval
# (as you know, neural networks prefer to deal with small input values).


train_datagen = ImageDataGenerator(rescale=1./255) # Rescales all images by 1/255
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150), # Resizes all images to 150 × 150
    batch_size=20,
    class_mode='binary') # Because you use binary_crossentropy loss, you need binary labels

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

# Read details about fit_generator, details right before the "Listing 5.8" in book


# Fit Model

history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=30,
      validation_data=validation_generator,
      validation_steps=50)

# Save your model for future use
model.save('cats_and_dogs_small_1.h5')

# Validate Model by Plotting curve to investigate

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

# These plots are characteristic of over fitting.
# The training accuracy increases linearly over time, until it reaches nearly 100%,
# whereas the validation accuracy stalls at 70–72%. The validation loss reaches its
# minimum after only five epochs and then stalls, whereas the training loss keeps
# decreasing linearly until it reaches nearly 0.

# Because you have relatively few training samples (2,000),
# over fitting will be your number-one concern.
# You already know about a number of techniques that can help mitigate over fitting,
# such as dropout and weight decay (L2 regularization). We’re now going to work with
# a new one, specific to computer vision and used almost universally when processing
# images with deep-learning models: data augmentation.



