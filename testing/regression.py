import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# linear regression mode: ŷ = w * x + b
# w and b are variables, x is the input which we can define as a placeholder
# We use the Mean Squared Error to formulate a cost function where y is a placeholder as well
# input x: tf_x defined as a placeholder
# input y: tf_y defined as a placeholder
# model parameter w: weight defined as a variable
# model parameter b: bias defined as a variable
# model output ŷ: y_hat returned by TF operations to compute the prediction using the model
g = tf.Graph()
with g.as_default():
    tf.set_random_seed(123)
    tf_x = tf.placeholder(shape=(None), dtype=tf.float32, name='tf_x')
    tf_y = tf.placeholder(shape=(None), dtype=tf.float32, name='tf_y')
    weight = tf.Variable(tf.random_normal(shape=(1, 1), stddev=0.25), name='weight')
    bias = tf.Variable(0.0, name='bias')
    y_hat = tf.add(weight * tf_x, bias, name='y_hat')
    cost = tf.reduce_mean(tf.square(tf_y - y_hat), name='cost')
    optim = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optim.minimize(cost, name='train_op')

np.random.seed(0)
def make_random_data():
    x = np.random.uniform(low=-2, high=4, size=200)
    y = []
    for t in x:
        r = np.random.normal(loc=0.0, scale=(0.5 + t*t/3), size=None)
        y.append(r)
    return x, 1.726*x - 0.84 + np.array(y)
x, y = make_random_data()
plt.plot(x, y, 'o')
plt.show()

# Now we train the model
x_train, y_train = x[:100], y[:100]
x_test, y_test = x[100:], y[100:]
n_epochs = 500
training_costs = []
with tf.Session(graph=g) as sess:
    sess.run(tf.global_variables_initializer())
    for e in range(n_epochs):
        c, _ = sess.run([cost, train_op], feed_dict={tf_x: x_train, tf_y: y_train})
        training_costs.append(c)
        if not e % 50:
            print('Epoch {}: {}'.format(e, c))
plt.plot(training_costs)
plt.show()
