import tensorflow as tf
import numpy as np

# Linear regression model. 2 placeholders for x and y. Variables for weights and bias.
# The model is z = w * x + b, we use the cost function Mean Squared Error.
# We learn the weights using the gradient descent optimizer

class TfLinreg(object):
    def __init__(self, x_dim, learning_rate=0.01, random_seed=None):
        self.x_dim = x_dim
        self.learning_rate = learning_rate
        self.g = tf.Graph()
        with self.g.as_default():
            tf.set_random_seed(random_seed)
            self.build()
            self.init_op = tf.global_variables_initializer()
    
    def build(self):
        self.X = tf.placeholder(dtype=tf.float32, shape=(None, self.x_dim), name='x_input')
        self.y = tf.placeholder(dtype=tf.float32, shape=(None), name='y_input')
        w = tf.Variable(tf.zeros(shape=(1)), name='weight')
        b = tf.Variable(tf.zeros(shape=(1)), name='bias')
        self.z_net = tf.squeeze(w * self.X + b, name='z_net')
        sqr_errors = tf.square(self.y - self.z_net, name='sqr_errors')
        self.mean_cost = tf.reduce_mean(sqr_errors, name='mean_cost')
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate, name='GradientDescent')
        self.optimizer = optimizer.minimize(self.mean_cost)

    def train(self, X_train, y_train, num_epochs=10):
        self.sess = tf.Session(graph = self.g)
        self.sess.run(self.init_op)
        training_costs = []
        for i in range(num_epochs):
            _, cost = self.sess.run([self.optimizer, self.mean_cost], feed_dict={self.X:X_train, self.y:y_train})
            training_costs.append(cost)
        return training_costs

    def predict(self, X_test):
        return self.sess.run(self.z_net, feed_dict={self.X:X_test})
