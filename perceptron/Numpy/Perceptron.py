import tensorflow
import numpy as np
import matplotlib.pyplot as plt
import os
from PerceptronNumpy import perceptron_train
from PerceptronNumpy import perceptron_predict

# We extract the data, split it into data and classes
data = np.genfromtxt('../perceptron_toydata.txt', delimiter='\t')
X, y = data[:, :2], data[:, 2]
y = y.astype(np.int)
print('Class label counts:', np.bincount(y))

plt.scatter(X[y==0, 0], X[y==0, 1], label='Class 0', marker='o')
plt.scatter(X[y==1, 0], X[y==1, 1], label='Class 1', marker='s')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

# We shuffle the data and split it into a train and test set
shuffle_idx = np.arange(y.shape[0])
shuffle_rng = np.random.RandomState(123)
shuffle_rng.shuffle(shuffle_idx)
X, y = X[shuffle_idx], y[shuffle_idx]
X_train, X_test = X[shuffle_idx[:70]], X[shuffle_idx[70:]]
y_train, y_test = y[shuffle_idx[:70]], y[shuffle_idx[70:]]

# We standardize the training set
mu, sigma = X_train.mean(axis=0), X_train.std(axis=0)
X_train = (X_train - mu) / sigma
X_test = (X_test - mu) / sigma

# Now that the preprocessing is done, we check the dataset
plt.scatter(X_train[y_train==0, 0], X_train[y_train==0, 1], label='class 0', marker='o')
plt.scatter(X_train[y_train==1, 0], X_train[y_train==1, 1], label='class 1', marker='s')
plt.xlabel('feature 1')
plt.ylabel('feature 2')
plt.legend()
plt.show()


# We train the perceptron for 2 epochs
model_params = perceptron_train(X_train, y_train, mparams=None, zero_weights=True)
for _ in range(2):
	_ = perceptron_train(X_train, y_train, mparams=model_params)

# We compute the training and test error
train_errors = np.sum(perceptron_predict(X_train, model_params) != y_train)
test_errors = np.sum(perceptron_predict(X_test, model_params) != y_test)
print('Number of training errors', train_errors)
print('Number of test errors', test_errors)

# Perceptron is a linear function with threshold: w1*x1 + w2*x2 + b >= 0
# Equation can be rearranged to become: - w1*x1/w2 - b/w2 <= x2
x_min = -2
y_min = ( -(model_params['weights'][0] * x_min) / model_params['weights'][1]
          -(model_params['bias'] / model_params['weights'][1]) )
x_max = 2
y_max = ( -(model_params['weights'][0] * x_max) / model_params['weights'][1]
          -(model_params['bias'] / model_params['weights'][1]) )
fig, ax = plt.subplots(1, 2, sharex=True, figsize=(7, 3))
ax[0].plot([x_min, x_max], [y_min, y_max])
ax[1].plot([x_min, x_max], [y_min, y_max])
ax[0].scatter(X_train[y_train==0, 0], X_train[y_train==0, 1], label='class 0', marker='o')
ax[0].scatter(X_train[y_train==1, 0], X_train[y_train==1, 1], label='class 1', marker='s')
ax[1].scatter(X_test[y_test==0, 0], X_test[y_test==0, 1], label='class 0', marker='o')
ax[1].scatter(X_test[y_test==1, 0], X_test[y_test==1, 1], label='class 1', marker='s')
ax[1].legend(loc='upper left')
plt.show()
