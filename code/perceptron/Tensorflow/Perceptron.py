import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

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

# We setup the perceptron graph
g = tf.Graph()
n_features = X_train.shape[1]
with g.as_default() as g:
	features = tf.placeholder(dtype=tf.float32, shape=[None, n_features], name='features')
	targets = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='targets')
	params = {
		'weights': tf.Variable(
			tf.zeros(shape=[n_features, 1], dtype=tf.float32), 
			name='weights'),
		'bias': tf.Variable([[0.]], dtype=tf.float32, name='bias')
	}
	# forward pass
	linear = tf.matmul(features, params['weights']) + params['bias']
	ones = tf.ones(shape=tf.shape(linear))
	zeros = tf.zeros(shape=tf.shape(linear))
	prediction = tf.where(tf.less(linear, 0), zeros, ones, name='prediction')
	# weight update
	diff = targets - prediction
	weight_update = tf.assign_add(params['weights'], tf.reshape(diff * features, (n_features, 1)))
	bias_update = tf.assign_add(params['bias'], diff)
	saver = tf.train.Saver()

# We train the perceptron for 5 training samples
with tf.Session(graph=g) as sess:
	sess.run(tf.global_variables_initializer())
	i = 0
	for example, target in zip(X_train, y_train):
		feed_dict = {
			features: example.reshape(-1, n_features),
			targets: target.reshape(-1, 1)
		}
		_, _ = sess.run([weight_update, bias_update], feed_dict=feed_dict)
		i += 1
		if i >= 4:
			break
	modelparams = sess.run(params)
	print('Model parameters:\n', modelparams)
	saver.save(sess, save_path='perceptron')
	pred = sess.run(prediction, feed_dict={features: X_train})
	errors = np.sum(pred.reshape(-1) != y_train)
	print('Number of training errors:', errors)

# We can continue training after restoring the session from a local checkpoint
with tf.Session(graph=g) as sess:
	saver.restore(sess, os.path.abspath('perceptron'))
	for epoch in range(1):
		for example, target in zip(X_train, y_train):
			feed_dict = {
				features: example.reshape(-1, n_features),
				targets: target.reshape(-1, 1)
			}
			_, _ = sess.run([weight_update, bias_update], feed_dict=feed_dict)
			modelparams = sess.run(params)
	saver.save(sess, save_path='perceptron')
	pred = sess.run(prediction, feed_dict={features: X_train})
	train_errors = np.sum(pred.reshape(-1) != y_train)
	pred = sess.run(prediction, feed_dict={features: X_train})
	test_errors = np.sum(pred.reshape(-1) != y_train)
	print('Number of training errors:', train_errors)
	print('Number of test errors:', test_errors)

# We can restore the session from a meta graph

with tf.Session() as sess:
	saver = tf.train.import_meta_graph(os.path.abspath('perceptron.meta'))
	saver.restore(sess, os.path.abspath('perceptron'))
	pred = sess.run('prediction:0', feed_dict={'features:0': X_train})
	train_errors = np.sum(pred.reshape(-1) != y_train)
	pred = sess.run('prediction:0', feed_dict={'features:0': X_test})
	test_errors = np.sum(pred.reshape(-1) != y_test)
	print('Number of training errors', train_errors)
	print('Number of test errors', test_errors)
