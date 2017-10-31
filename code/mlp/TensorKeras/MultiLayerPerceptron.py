import os
import struct
import numpy as np
import tensorflow as tf
import tensorflow.contrib.keras as keras

def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte' % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
        images = ((images / 255.) - .5) * 2
    return images, labels

# We load the training and testing sets
X_train, y_train = load_mnist('../mnist/', kind='train')
X_test, y_test = load_mnist('../mnist/', kind='t10k')
print('Rows: {}, Columns: {}'.format(X_train.shape[0], X_train.shape[1]))
print('Rows: {}, Columns: {}'.format(X_test.shape[0], X_test.shape[1]))

# Mean centering and normalization
mean_vals = np.mean(X_train, axis=0)
std_val = np.std(X_train)
X_train_centered = (X_train - mean_vals) / std_val
X_test_centered = (X_test - mean_vals) / std_val
del X_train, X_test

# We set the random seed for numpy and tensorflow to have consistent results
random_seed = 123
np.random.seed(random_seed)
tf.set_random_seed(random_seed)
y_train_onehot = keras.utils.to_categorical(y_train)

# We create the MLP wuth three layers, first two habe 50 hidden units with tanh activation
# The last layer has 10 units for the 10 class labels and uses softmax to give probability
# First we initialize the model with the Sequential class for a feedforward neural network
# Then we add the layers we want, first input_dim is the number of features.
# The number of output units and input_units of 2 consecutive layers have to match.
# The number of units in the output layer is equal to the number of unique class labels.
# We use Glorot initialization for weight matrices, a more robut way to initialize.
# The categorical cross-entropy is the cost function in multiclass via softmax function.
model = keras.models.Sequential()
model.add(
    keras.layers.Dense(
        units=50,
        input_dim=X_train_centered.shape[1],
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        activation='tanh'))
model.add(
    keras.layers.Dense(
        units=50,
        input_dim=50,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        activation='tanh'))
model.add(
    keras.layers.Dense(
        units=y_train_onehot.shape[1],
        input_dim=50,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        activation='softmax'))
sgd_optimizer = keras.optimizers.SGD(lr=0.001, decay=1e-7, momentum=.9)
model.compile(optimizer=sgd_optimizer, loss='categorical_crossentropy')
# We can follow the optimization of the cost function with verbose=1
history = model.fit(X_train_centered, y_train_onehot, batch_size=64, epochs=50, verbose=1, validation_split=0.1)

# We predict using the model
y_train_pred = model.predict_classes(X_train_centered, verbose=0)
correct_preds = np.sum(y_train == y_train_pred, axis=0)
train_acc = correct_preds / y_train.shape[0]
print('Training accuracy: {}%'.format(train_acc * 100))
y_test_pred = model.predict_classes(X_test_centered, verbose=0)
correct_preds = np.sum(y_test == y_test_pred, axis=0)
test_acc = correct_preds / y_test.shape[0]
print('Test accuracy: {}%'.format(test_acc * 100))
