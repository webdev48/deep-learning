from keras.datasets import boston_housing
from keras import models
from keras import layers
import numpy as np

# Step.0 : Investigate Data

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

print(train_data.shape)  # very small data set (404, 13)
print(test_data.shape)  # (102, 13) >> 13 means 13 numerical features


# Step.1 : Prepare Data

print(train_targets)  # Median value of owner occupied homes, in thousands

# It would be problematic to feed into a neural network values that all take wildly
# different ranges. The network might be able to automatically adapt to such heterogeneous
# data, but it would definitely make learning more difficult.
# A widespread best practice to deal with such data is to do feature-wise normalization:
# for each feature in the input data (a column in the input data matrix),
# you subtract the mean of the feature and divide by the standard deviation,
# so that the feature is centered around 0 and has a unit standard deviation.


mean = train_data.mean(axis=0)
train_data = train_data - mean
std = train_data.std(axis=0)
train_data = train_data/std

test_data = test_data - mean
test_data = test_data/std

# Step.2 : Build Your Network


def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model


# Note that you compile the network with the mse loss function—mean squared error,
# the square of the difference between the predictions and the targets.This is a widely
# used loss function for regression problems.

# You’re also monitoring a new metric during training: mean absolute error (MAE).
# It’s the absolute value of the difference between the predictions and the targets.
# For instance, an MAE of 0.5 on this problem would mean your predictions
# are off by $500 on average

# The network ends with a single unit and no activation (it will be a linear layer).
#  This is a typical setup for scalar regression (a regression where you’re trying to
# predict a single continuous value). Applying an activation function would constrain
# the range the output can take; for instance, if you applied a sigmoid activation function
#  to the last layer, the network could only learn to predict values between 0 and 1.
# Here, because the last layer is purely linear, the network is free to learn to
# predict values in any range.

# Step.3 : Validating your approach using K-fold

# Explanation on page.87 ( for very small data set, we split data in partitions , fold 1/2/3 )

k = 4
num_val_samples = len(train_data) // k
num_epochs = 100
all_scores = []

for i in range(k):
    print('processing fold #', i)
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate([train_data[:i * num_val_samples], train_data[(i + 1) * num_val_samples:]],
                                        axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples], train_targets[(i + 1) * num_val_samples:]], axis=0)

    model = build_model()

# Step.5 : Fit Model

    model.fit(
        partial_train_data,
        partial_train_targets,
        epochs=num_epochs,
        batch_size=1,
        verbose=0
    )

# Step.6 : Evaluate Model
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)

# Step.7 : Check results

    # [1.8194324144042364, 2.3356197281639175, 2.647253158068893, 2.3047475708593237]
    print(all_scores)

    # Looks like we are still off on average by 2.25 = $2250


# Differences between MAE and RMSE :
# https://medium.com/human-in-a-machine-world/mae-and-rmse-which-metric-is-better-e60ac3bde13d


# Step.8 : Plot graph for validation logs at each fold ( page 88/89 )

# Another technique to learn on page 89 is that
# if plots are difficult to read due to scale issues and relatively high variance,

# then:

# replace each point with an exponential moving average of the previous points to obtain a smooth curve

# Step.9 :
# Find the epoch point where error stops decreasing
# Retrain the model with the newly find optimal epoch


# Wrap up

# Regression is done using different loss functions than what we used for classification.
# Mean squared error (MSE) is a loss function commonly used for regression.

# Similarly, evaluation metrics to be used for regression differ from those used for classification;
# naturally, the concept of accuracy does not apply for regression.
# A common regression metric is mean absolute error (MAE).


# When features in the input data have values in different ranges,
# each feature should be scaled independently as a pre processing step.

# When there is little data available, using K-fold validation is a great way to reliably evaluate a model.

# When little training data is available, it’s preferable to use a small network with few hidden layers
# (typically only one or two), in order to avoid severe overfitting.
