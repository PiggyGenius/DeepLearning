import os
import struct
import numpy as np
import tensorflow as tf
import cnn

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

# We load the data and normalize the data, on all features to avoid constant values and division by 0
X_data, y_data = load_mnist('./mnist/', kind='train')
X_test, y_test = load_mnist('./mnist/', kind='t10k')
X_train, y_train = X_data[:50000,:], y_data[:50000]
X_valid, y_valid = X_data[50000:,:], y_data[50000:]
mean_vals = np.mean(X_train, axis=0)
std_val = np.std(X_train)
X_train_centered = (X_train - mean_vals) / std_val
X_valid_centered = (X_valid - mean_vals) / std_val
X_test_centered = (X_test - mean_vals) / std_val
print('Training:   ', X_train.shape, y_train.shape)
print('Validation: ', X_valid.shape, y_valid.shape)
print('Test Set:   ', X_test.shape, y_test.shape)

# We create and train our CNN
random_seed = 123
learning_rate = 1e-4
g = tf.Graph()
with g.as_default():
    tf.set_random_seed(random_seed)
    cnn.build(learning_rate)
    saver = tf.train.Saver()
with tf.Session(graph=g) as sess:
    cnn.train(sess, training_set=(X_train_centered, y_train),
            validation_set=(X_valid_centered, y_valid), initialize=True, random_seed=123)
    cnn.save(saver, sess, epoch=20)

# We can restore the model
# del g
random_seed = 123
learning_rate = 1e-4
g2 = tf.Graph()
with g2.as_default():
    tf.set_random_seed(random_seed)
    cnn.build(learning_rate)
    saver = tf.train.Saver()
with tf.Session(graph=g2) as sess:
    cnn.load(saver, sess, epoch=20, path='./model')
    preds = cnn.predict(sess, X_test_centered, return_proba=False)
    print('Test accuracy: {}'.format(100 * np.sum(preds == y_test) / len(y_test)))

# We can look at the probabilities of the label
np.set_printoptions(precision=2, suppress=True)
with tf.Session(graph=g2) as sess:
    cnn.load(saver, sess, epoch=20, path='./model/')
    print(cnn.predict(sess, X_test_centered[:10], return_proba=False))
    print(cnn.predict(sess, X_test_centered[:10], return_proba=True))

# We can retrain the model 20 more epochs
with tf.Session(graph=g2) as sess:
    cnn.load(saver, sess, epoch=20, path='./model/')
    cnn.train(sess, training_set=(X_train_centered, y_train), 
            validation_set=(X_valid_centered, y_valid), initialize=False, epochs=20, random_seed=123)
    cnn.save(saver, sess, epoch=40, path='./model/')
    preds = cnn.predict(sess, X_test_centered, return_proba=False)
    print('Test accuracy: {}'.format(100 * np.sum(preds == y_test) / len(y_test)))
